"""讀 PixInsight 的 XISF（XISF 1.0 規格：https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html）。

檔案結構：8 bytes 簽名 "XISF0100" → uint32 標頭長度 → uint32 保留 → UTF-8 的 XML 標頭 → 資料區塊。
<Image> 的 location="attachment:位置:長度" 指到資料；geometry="寬:高:色版數"；預設 Planar（一個色版一整塊），
第 0 列是畫面最上面（跟記憶體裡的慣例相同，不用翻）。壓縮：zlib／lz4／lz4hc／zstd，可加 +sh（byte shuffle）。

只讀第一張影像；FITS 關鍵字轉成 astropy 的 Header（FILTER、OBJECT…），其他模組照原本的方式讀。
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

SIGNATURE = b"XISF0100"
NS = "{http://www.pixinsight.com/xisf}"
DTYPES = {"UInt8": np.uint8, "UInt16": np.uint16, "UInt32": np.uint32, "UInt64": np.uint64,
          "Float32": np.float32, "Float64": np.float64}


class XISFError(ValueError):
    pass


@dataclass
class XISFImage:
    data: np.ndarray                 # (C, H, W) 或 (H, W)，原本的型別
    sample_format: str
    bounds: tuple[float, float] | None
    color_space: str
    keywords: list[tuple[str, str, str]] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)

    def header(self) -> fits.Header:
        h = fits.Header()
        for name, value, comment in self.keywords:
            if not name or name in ("COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                                    "NAXIS3", "EXTEND", "END"):
                continue
            v: object = value.strip()
            if isinstance(v, str) and v.startswith("'") and v.endswith("'"):
                v = v[1:-1].strip()
            else:
                try:
                    v = int(v) if v.lstrip("+-").isdigit() else float(v)
                except (TypeError, ValueError):
                    if v in ("T", "F"):
                        v = v == "T"
            try:
                h[name[:8]] = (v, comment[:60])
            except (ValueError, KeyError):
                pass
        return h


def _decompress(raw: bytes, spec: str, item_size: int) -> bytes:
    parts = spec.split(":")
    codec = parts[0]
    expected = int(parts[1]) if len(parts) > 1 and parts[1] else None
    shuffled = codec.endswith("+sh")
    codec = codec.replace("+sh", "")
    if codec == "zlib":
        out = zlib.decompress(raw)
    elif codec in ("lz4", "lz4hc"):
        try:
            import lz4.block
        except ImportError as exc:
            raise XISFError("這個 XISF 用 LZ4 壓縮，需要 lz4 套件（pip install lz4）") from exc
        out = lz4.block.decompress(raw, uncompressed_size=expected)
    elif codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise XISFError("這個 XISF 用 Zstandard 壓縮，需要 zstandard 套件（pip install zstandard）") from exc
        out = zstandard.ZstdDecompressor().decompress(raw, max_output_size=expected or 0)
    else:
        raise XISFError(f"不支援的 XISF 壓縮方式：{codec}")
    if shuffled:
        size = int(parts[2]) if len(parts) > 2 else item_size
        n = len(out) // size
        arr = np.frombuffer(out[:n * size], np.uint8).reshape(size, n).T.reshape(-1)
        out = arr.tobytes() + out[n * size:]
    return out


def read_header(path: Path) -> tuple[ET.Element, int]:
    with Path(path).open("rb") as fh:
        head = fh.read(16)
        if len(head) < 16 or head[:8] != SIGNATURE:
            raise XISFError(f"{Path(path).name} 不是 XISF 檔")
        length = struct.unpack("<I", head[8:12])[0]
        xml = fh.read(length)
    try:
        root = ET.fromstring(xml.split(b"\x00", 1)[0])
    except ET.ParseError as exc:
        raise XISFError(f"XISF 標頭讀不懂：{exc}") from exc
    return root, 16 + length


def read(path: Path | str) -> XISFImage:
    path = Path(path)
    root, _ = read_header(path)
    image = root.find(f"{NS}Image")
    if image is None:
        image = root.find("Image")
    if image is None:
        raise XISFError(f"{path.name} 裡沒有影像")
    geometry = [int(v) for v in image.get("geometry", "").split(":")]
    if len(geometry) < 2:
        raise XISFError("XISF 影像沒有尺寸")
    w, h = geometry[0], geometry[1]
    channels = geometry[2] if len(geometry) > 2 else 1
    fmt = image.get("sampleFormat", "UInt16")
    if fmt not in DTYPES:
        raise XISFError(f"不支援的 XISF 樣本格式：{fmt}")
    dtype = np.dtype(DTYPES[fmt])
    if image.get("byteOrder", "little") == "big":
        dtype = dtype.newbyteorder(">")
    loc = image.get("location", "").split(":")
    if not loc or loc[0] != "attachment" or len(loc) < 3:
        raise XISFError("只支援資料附在檔案裡的 XISF（attachment）")
    pos, size = int(loc[1]), int(loc[2])
    with path.open("rb") as fh:
        fh.seek(pos)
        raw = fh.read(size)
    if len(raw) < size:
        raise XISFError(f"{path.name} 不完整（資料比標頭寫的短）")
    if image.get("compression"):
        sub = image.get("subblocks")
        if sub:
            chunks, offset = [], 0
            for item in sub.split(":"):
                csize, _usize = (int(v) for v in item.split(","))
                chunks.append(_decompress(raw[offset:offset + csize], image.get("compression"), dtype.itemsize))
                offset += csize
            raw = b"".join(chunks)
        else:
            raw = _decompress(raw, image.get("compression"), dtype.itemsize)
    expected = w * h * channels * dtype.itemsize
    if len(raw) < expected:
        raise XISFError(f"{path.name} 的影像資料不完整")
    arr = np.frombuffer(raw[:expected], dtype)
    if image.get("pixelStorage", "Planar") == "Normal":
        arr = arr.reshape(h, w, channels).transpose(2, 0, 1)
    else:
        arr = arr.reshape(channels, h, w)
    arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    data = arr[0] if channels == 1 else arr
    bounds = None
    if image.get("bounds"):
        lo, hi = (float(v) for v in image.get("bounds").split(":"))
        bounds = (lo, hi)
    keywords = [(k.get("name", ""), k.get("value", ""), k.get("comment", ""))
                for k in list(image.iter(f"{NS}FITSKeyword")) + list(image.iter("FITSKeyword"))]
    props = {p.get("id", ""): p.get("value", p.text or "") for p in image.iter(f"{NS}Property")}
    return XISFImage(np.array(data), fmt, bounds, image.get("colorSpace", "Gray"), keywords, props)


def to_unit(img: XISFImage) -> np.ndarray:
    """換成 0–1 的 float32：整數除以型別上限；浮點照 bounds（沒寫就看資料，超過 1 才縮）。"""
    data = img.data
    if np.issubdtype(data.dtype, np.integer):
        return (data.astype(np.float32) / np.float32(np.iinfo(data.dtype).max)).astype(np.float32)
    out = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = img.bounds if img.bounds else (0.0, 1.0)
    if hi > lo and (lo, hi) != (0.0, 1.0):
        out = (out - np.float32(lo)) / np.float32(hi - lo)
    peak = float(out.max()) if out.size else 0.0
    if peak > 1.0:
        out /= np.float32(peak)
    return out
