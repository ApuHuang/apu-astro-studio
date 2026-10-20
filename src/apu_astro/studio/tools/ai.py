"""AI 降噪（Noise2Noise 訓練的 U-Net，ONNX 格式，用 DirectML 在 Windows 的顯示卡上跑，沒有就用 CPU）。

模型在使用者自己的資料上訓練：同一組對齊好的 sub 分兩半各自疊，學「同一片天空的兩份噪聲」之間的差，
不需要乾淨的標準答案（tools/ai/make_n2n.py、train_n2n.py）。
一個色版一個色版做；先換成「比天空亮幾倍噪聲」的單位，所以不同相機、不同張數共用同一個模型。

模型的修正量只有細的部分（輸出先減掉自己模糊 3 px 的版本，做在模型裡）：降噪該去掉的是細顆粒，
大範圍的亮度、顏色不該被改（第一版沒有這個限制，M81 的星系核心整片變暗變色）。大範圍的斑駁交給一般的降噪。
驗證組 M81（4 張 sub）：天空誤差 1.43σ → 0.74σ，亮處 3.73 → 3.40（也比原圖好）。
注意：降噪後量到的噪聲變小，「自動拉伸」會拉得更重，大範圍的斑駁跟著被拉出來；所以還沒放進建議流程。
guard＝推論時再做一次高通（舊模型用；現在的模型已經內建，再做反而變差）。
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import backend, ops
from . import T, Ctx, Tool, record_denoise_ratio, register, slider, toggle

MODEL = Path(__file__).resolve().parents[2] / "assets" / "n2n.onnx"
MODEL_RGB = Path(__file__).resolve().parents[2] / "assets" / "n2n_rgb.onnx"   # 彩色：三個色版一起看（有的話）
GUARD_SIGMA = 3.0
_session = None
_session_rgb = None
_session_lock = threading.Lock()


def available() -> bool:
    return MODEL.is_file()


def available_rgb() -> bool:
    return MODEL_RGB.is_file()


def _open(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), opts, providers=providers)


def session():
    global _session
    with _session_lock:
        if _session is None:
            _session = _open(MODEL)
    return _session


def session_rgb():
    global _session_rgb
    with _session_lock:
        if _session_rgb is None:
            _session_rgb = _open(MODEL_RGB)
    return _session_rgb


_noise_dist_cache: dict[str, int] = {}


def noise_dist() -> int:
    """模型訓練時量噪聲用相隔幾列的像素差（模型旁邊的 n2n.json；沒有＝1，相鄰像素，第八版）。
    推論要跟訓練用一樣的量法，不然噪聲單位不一樣（見 tools/ai/train_n2n.py 的 robust_sky_sigma）。"""
    key = str(MODEL)
    if key not in _noise_dist_cache:
        meta = MODEL.with_suffix(".json")
        try:
            _noise_dist_cache[key] = int(json.loads(meta.read_text(encoding="utf-8")).get("noise_dist", 1))
        except (OSError, ValueError):
            _noise_dist_cache[key] = 1
    return _noise_dist_cache[key]


def sky_sigma(a: np.ndarray, dist: int | None = None) -> tuple[float, float]:
    """天空與噪聲 σ。dist：量噪聲用相隔幾列的像素差；預設跟著目前的模型（評估工具比較不同模型時固定用 1）。"""
    sky = float(np.median(a[:: 4, :: 4]))
    dist = noise_dist() if dist is None else dist
    if dist == 1:
        d = (a[1::2, ::2] - a[:-1:2, ::2])[::2, ::2]
    else:
        d = (a[dist:, ::2] - a[:-dist, ::2])[::4, ::2]
    sigma = float(np.median(np.abs(d - np.median(d)))) * 1.4826 / math.sqrt(2)
    return sky, max(sigma, 1e-9)


DARK_LIMIT = 4.0      # 比周圍（5×5 中位數）暗這麼多倍局部噪聲的單點，先補掉
BRIGHT_FADE = 30.0    # 比天空亮這麼多倍噪聲的地方，AI 的修正漸漸減少（亮處噪聲本來就不明顯，模型在那裡也比較不準）


def _median5(x: np.ndarray) -> np.ndarray:
    """5×5 中位數；有 N 卡用 GPU（整張 2000 萬像素 0.02 秒，CPU 要 6 秒）。"""
    cp = backend.gpu() if x.size > 1_000_000 else None
    if cp is not None:
        try:
            import cupyx.scipy.ndimage as cnd
            return cp.asnumpy(cnd.median_filter(cp.asarray(x, dtype=cp.float32), size=5, mode="reflect"))
        except Exception:  # noqa: BLE001  GPU 出問題就用 CPU
            pass
    return ndimage.median_filter(x, size=5, mode="reflect")


def fix_dark_pixels(x: np.ndarray) -> tuple[np.ndarray, int]:
    """x：噪聲單位的 2D 影像。比周圍暗很多的單點（壞點、對齊／去馬賽克留下的）換成周圍的中位數。
    天文影像裡沒有真的「一個像素比四周暗 4σ」的東西；亮的點（星）不動。
    不補的話 AI 只修得掉一部分（修正量有上限），四周變乾淨後反而更明顯，拉伸、銳化後變成黑點。"""
    med = _median5(x)
    s = np.sqrt(1 + np.maximum(med, 0) / 16)
    bad = (x - med) < -DARK_LIMIT * s
    n = int(bad.sum())
    if n:
        x = np.where(bad, med, x).astype(np.float32)
    return x, n


def bright_fade(pred: np.ndarray) -> np.ndarray:
    """AI 修正量的權重：天空 1，亮度到 BRIGHT_FADE 倍噪聲時 0.5，再亮就漸漸到 0。
    模擬驗證（NGC6888 Ha、NGC2244）：不淡出時亮處、星核的誤差比原圖多 10～30%，淡出後跟原圖一樣，天空不受影響。
    亮度先模糊 6 px：權重在星點周圍慢慢變（5 px 方塊平均時權重在星點邊緣突然掉到 0，
    星點周圍的修正被切掉一半，留下 0.2～0.3σ 的低頻光暈）。"""
    level = ops.blur(np.maximum(pred, 0).astype(np.float32), 6.0)
    return (1.0 / (1.0 + (level / BRIGHT_FADE) ** 2)).astype(np.float32)


def star_protect(fixed: list[np.ndarray]) -> np.ndarray:
    """星點範圍的權重（星心 1、往外平滑到 0）。AI 對三個色版各自降噪，暗星的輪廓會被修得不一樣，
    星色就偏了（模擬測試：11～19% 的星比加噪的輸入還偏）；星點範圍不修，只修星點之間的背景。
    預設不用：星點周圍留著原本的噪聲、背景乾淨，每顆星像長了一圈噪聲（比星色偏一點更明顯）。"""
    from apu_processing import detail as pdetail
    L = np.mean(fixed, axis=0)
    stars = pdetail.find_stars(L, 4.0)
    wgt = ndimage.maximum_filter(stars.weight.astype(np.float32), size=3)
    return ops.blur(wgt, 1.0)


def _block_mean(x: np.ndarray, f: int) -> np.ndarray:
    h, w = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _up2(c: np.ndarray, axis: int) -> np.ndarray:
    """沿一個軸放大 2 倍（線性，像素中心對齊）：新的像素 2i＝¾c[i]＋¼c[i−1]、2i＋1＝¾c[i]＋¼c[i＋1]。"""
    c = np.moveaxis(c, axis, 0)
    prev = np.concatenate([c[:1], c[:-1]])
    nxt = np.concatenate([c[1:], c[-1:]])
    out = np.empty((2 * c.shape[0],) + c.shape[1:], np.float32)
    out[0::2] = 0.75 * c + 0.25 * prev
    out[1::2] = 0.75 * c + 0.25 * nxt
    return np.moveaxis(out, 0, axis)


def _upsample(c: np.ndarray, f: int, shape: tuple[int, int]) -> np.ndarray:
    """_block_mean 的反向（雙線性，像素中心對齊）。"""
    h, w = shape
    if f == 2:          # 常用的情況：直接算，比 map_coordinates 快十倍（完整解析度三色省 3 秒）
        up = _up2(_up2(c.astype(np.float32), 0), 1)
        return np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")[:h, :w]
    yy = np.clip((np.arange(h, dtype=np.float32) + 0.5) / f - 0.5, 0, c.shape[0] - 1)
    xx = np.clip((np.arange(w, dtype=np.float32) + 0.5) / f - 0.5, 0, c.shape[1] - 1)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(c, [Y, X], order=1).astype(np.float32)


COARSE = (2,)        # 完整解析度的 AI 之後，縮小幾倍再跑一次（處理斑駁；見 denoise_planes）
COARSE_LIMIT = 1.0   # 縮小那層每個像素最多改幾倍噪聲（斑駁只有 0.2～0.5σ）
COARSE_STRUCT = 3.0  # 縮小那層：中尺度的起伏超過斑駁典型起伏的幾倍就算結構，不改（見 denoise_planes）


def denoise_planes(xs: list[np.ndarray], amount: float = 1.0, ctx: Ctx | None = None,
                   protect_stars: bool = False, coarse: tuple[int, ...] = (),
                   chroma: float = 0.0, rgb: bool | None = None) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """一組色版（各自的噪聲單位）完整的 AI 降噪：補暗點 → 模型 → 亮處淡出 → 去掉淡出造成的低頻。
    淡出權重三色共用（用三色平均的亮度）：各自算的話，偏黃的星系裡紅色版淡出得多、藍色版修得多，星點會變色。
    coarse：再縮小這幾倍各跑一次，只取修正量放大加回去（亮處照完整解析度的亮度淡出）。模型只看得到幾個像素的
    噪聲，4 px 以上的斑駁只減 15%；縮小一半後斑駁變成細顆粒，同一個模型就處理得到。M81 實拍驗證（4 張 vs 另外
    20 張的平均）：斑駁 0.166 → 0.142σ，比後面再接 Processing 降噪（0.154）好，暗星也保留得比較多（0.92 vs 0.88）；
    縮小 4 倍那層暗星掉太多（0.83），不用。細節見 _coarse_pass。chroma：色彩降噪模糊幾 px（見 smooth_chroma）。
    回傳 (補過暗點的輸入, 結果, 補了幾個暗點)。後製工作室與 tools/ai/artifact_audit.py 都用這個。"""
    fixed, n_dark = [], 0
    for x in xs:
        f, n = fix_dark_pixels(x)
        fixed.append(f)
        n_dark += n
    if rgb is None:
        rgb = available_rgb()
    if rgb and len(fixed) == 3:
        # 彩色模型：三色一起看（分得出三色一起亮的結構和只有一色的噪聲）
        preds = list(denoise_rgb(np.stack(fixed), ctx))
    else:
        preds = []
        for i, x in enumerate(fixed):
            preds.append(denoise_plane(x, ctx))
            if ctx is not None:
                ctx.progress((i + 1) / len(fixed), "ai")
    w = bright_fade(np.mean(preds, axis=0))
    if protect_stars:
        w = w * (1 - star_protect(fixed))
    outs = []
    for x, pred in zip(fixed, preds):
        fix = (pred - x) * w
        # 淡出後修正量又有一點低頻（星點周圍）；在權重高的地方再高通一次，星點核心（權重低）不動
        fix = fix - ops.blur(fix.astype(np.float32), GUARD_SIGMA) * w
        outs.append((x + fix * amount).astype(np.float32))
    for f in coarse:
        outs = _coarse_pass(outs, f, amount, ctx)
    if chroma > 0 and len(outs) >= 3:
        outs = smooth_chroma(outs, chroma)
    return fixed, outs, n_dark


def _coarse_pass(outs: list[np.ndarray], f: int, amount: float, ctx: Ctx | None) -> list[np.ndarray]:
    """縮小 f 倍再跑一次模型，只取修正量放大加回去（去斑駁）。
    1. 星點先「仿製填滿」（調和內插補掉）再給模型看：縮小後星點只剩 1～2 px，模型會把它當噪聲削掉、旁邊留下暗環
       （IC434）。以前改成星點附近不修，結果星點附近留著原本的斑、遠處很平滑，一塊塊交錯（LBN576，使用者：「星點周邊
       有點斑駁」）。填掉之後星點附近的斑也修得到，星點本身由完整解析度的亮度淡出保護。
       星點範圍（外擴 2 px）裡的修正量不用模型的（填平的平台模型會去壓），改用周圍修正量的「平均」（模糊 3 px，
       只算範圍外的）：修正量大部分是每個像素都不一樣的細噪聲（每色版 ±0.3σ），直接把邊上的值延伸進去等於把幾個
       隨機的值抹成一塊，每顆星旁邊一塊藍紫、綠色的斑（LBN576）。大星點中間離周圍太遠的地方就不修
    2. 修正量限制 ±COARSE_LIMIT（斑駁只有 0.2～0.5σ）
    3. 避開有結構的地方：縮小後暗帶變窄，模型把它當暗的噪聲斑填掉（NGC2244 暗帶模糊 6 px 後 −0.8σ → −0.5σ）。
       在填掉星點的縮小圖上量中尺度（完整尺寸 6 px 對 24 px）的起伏，周圍 12 px 取最大（邊緣正中間是 0），
       超過斑駁典型起伏 COARSE_STRUCT 倍就不改"""
    from apu_processing import detail as pdetail

    from . import stars as tstars
    small = np.stack([_block_mean(x, f) for x in outs]).astype(np.float32)
    units = [sky_sigma(c) for c in small]                      # 噪聲用沒填的量（填過的地方沒有噪聲，會量得太小）
    # 只填星核：每顆星自己的範圍（亮度降到噪聲以下為止，暗星 2～6 px）。用星點遮罩（含外圍淡淡的一圈）的話每顆星
    # 周圍 10～16 px 都被填平，模型看不到那裡的斑，密集星場星點旁邊一塊塊沒修到（LBN576）
    Ls = small.mean(axis=0)
    found = pdetail.find_stars(Ls, 3.0)   # 縮小後暗星的峰值掉得比噪聲多，門檻放低（多填一些沒關係）
    holes = tstars.star_footprints(Ls, found, float(f), 1.0)[0] if len(found.y) else np.zeros(Ls.shape, bool)
    filled = backend.harmonic_fill(small, holes) if holes.any() else small
    Lf = filled.mean(axis=0)
    band = ops.blur(Lf, 6.0 / f) - ops.blur(Lf, 24.0 / f)
    typical = 1.4826 * float(np.median(np.abs(band[::2, ::2] - np.median(band[::2, ::2])))) + 1e-6
    struct = ndimage.maximum_filter(np.abs(band), size=2 * max(1, int(round(6.0 / f))) + 1)
    gate = 1.0 / (1.0 + (struct / (COARSE_STRUCT * typical)) ** 4)
    w = bright_fade(np.mean(outs, axis=0))
    near = ndimage.binary_dilation(holes, iterations=2) if holes.any() else holes
    fixes = np.stack([(denoise_plane((c - sky) / max(sig, 1e-12), ctx) * sig + sky - c) * gate
                      for c, (sky, sig) in zip(filled, units)])
    # 修正量只留斑駁的尺度（縮小圖上模糊 1 px＝完整尺寸 2 px；更細的噪聲完整解析度那次已經處理），星點範圍外平均，
    # 星點範圍裡用調和內插平順地補進去：補的是平滑的斑駁修正量，星點旁邊的斑也修得到（以前用周圍的平均，周圍大多是
    # 細噪聲、平均起來幾乎是 0，等於星點旁邊沒修，其他地方都乾淨了，星旁邊剩下的暗斑看起來像黑圈：使用者 2026-10-19）
    valid = (~near).astype(np.float32)
    wsum = ops.blur(valid, 1.0)
    for k in range(len(fixes)):
        fixes[k] = ops.blur(fixes[k] * valid, 1.0) / np.maximum(wsum, 1e-3)
    inside = wsum < 0.3
    if inside.any():
        fixes = backend.harmonic_fill(fixes, inside)
    nxt = []
    for x, fx in zip(outs, fixes):
        fix = np.clip(_upsample(fx, f, x.shape), -COARSE_LIMIT, COARSE_LIMIT)
        nxt.append((x + fix * w * amount).astype(np.float32))
    return nxt


CHROMA_SIGMA = 1.5   # 色彩降噪：色彩差異模糊幾 px（完整尺寸）
CHROMA_FADE = 3.0    # 色彩降噪：比天空亮這麼多倍噪聲的地方只做一半（星點、星雲保留顏色；10 的時候星色變差的星 6% → 20%）


def smooth_chroma(outs: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    """色彩降噪：每個色版跟三色平均的差（色彩）模糊 sigma px，三色平均（亮度）不動。AI 每個色版各自降噪，剩下的噪聲
    不一樣，背景變平滑之後一顆顆綠色、紅色的小點很明顯（LBN576，使用者：「背景有點人造物綠斑」）。顏色細節本來就比
    亮度細節少，背景的色彩模糊一點看不出來；亮的地方（星點、星雲）漸漸不做，保留星色。"""
    L = np.mean(outs, axis=0).astype(np.float32)
    # 亮度用附近 5 px 內最亮的值：星點整顆（含外圍）都算亮，不會只保護到星心
    level = ops.blur(ndimage.maximum_filter(np.maximum(L, 0), size=5), 1.0)
    wgt = 1.0 / (1.0 + (level / CHROMA_FADE) ** 2)
    out = []
    for o in outs:
        c = (o - L).astype(np.float32)
        out.append((o + (ops.blur(c, sigma) - c) * wgt).astype(np.float32))
    return out


def denoise_rgb(x: np.ndarray, ctx: Ctx | None = None, tile: int = 640, pad: int = 96) -> np.ndarray:
    """彩色模型：x 是 (3, H, W)、每色已換成自己的噪聲單位。分塊方式跟 denoise_plane 一樣（沒有接縫）。"""
    sess = session_rgb()
    _c, h, w = x.shape
    out = np.empty_like(x, np.float32)
    n_tiles = ((h + tile - 1) // tile) * ((w + tile - 1) // tile)
    done = 0
    for y in range(0, h, tile):
        for xx in range(0, w, tile):
            if ctx is not None:
                ctx.check()
            y0, x0 = max(0, y - pad), max(0, xx - pad)
            y1, x1 = min(h, y + tile + pad), min(w, xx + tile + pad)
            block = np.ascontiguousarray(x[:, y0:y1, x0:x1], np.float32)
            ph, pw = (-block.shape[1]) % 8, (-block.shape[2]) % 8
            if ph or pw:
                block = np.pad(block, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            res = sess.run(None, {"x": block[None]})[0][0]
            th, tw = min(tile, h - y), min(tile, w - xx)
            out[:, y:y + th, xx:xx + tw] = res[:, y - y0:y - y0 + th, xx - x0:xx - x0 + tw]
            done += 1
            if ctx is not None:
                ctx.progress(done / n_tiles, "ai")
    return out


def denoise_plane(x: np.ndarray, ctx: Ctx | None = None, tile: int = 640, pad: int = 96,
                  guard: bool = False) -> np.ndarray:
    """x：已換成噪聲單位的 2D 影像。分塊跑，每塊四周多 pad 像素：96 px 比模型看得到的範圍大，
    錯開分塊位置算出來完全一樣（32 px 時富士山、IC434、M31 有 0.3～0.8σ 的接縫）。guard：修正量只留細的部分。"""
    sess = session()
    h, w = x.shape
    out = np.empty_like(x, np.float32)
    for y in range(0, h, tile):
        for xx in range(0, w, tile):
            if ctx is not None:
                ctx.check()
            y0, x0 = max(0, y - pad), max(0, xx - pad)
            y1, x1 = min(h, y + tile + pad), min(w, xx + tile + pad)
            block = np.ascontiguousarray(x[y0:y1, x0:x1], np.float32)
            # U-Net 下採樣三次：尺寸補到 8 的倍數
            ph, pw = (-block.shape[0]) % 8, (-block.shape[1]) % 8
            if ph or pw:
                block = np.pad(block, ((0, ph), (0, pw)), mode="reflect")
            res = sess.run(None, {"x": block[None, None]})[0][0, 0]
            th, tw = min(tile, h - y), min(tile, w - xx)
            out[y:y + th, xx:xx + tw] = res[y - y0:y - y0 + th, xx - x0:xx - x0 + tw]
    if guard:
        fix = x - out
        out = x - (fix - ops.blur(fix.astype(np.float32), GUARD_SIGMA))
    return out


def detail_noise(planes: np.ndarray) -> float:
    """跟 apu_processing.detail.find_stars 一樣的噪聲量法：亮度減掉 starlet 前三層以外的部分，取 MAD。"""
    from apu_processing import denoise as pdenoise
    L = planes.mean(axis=0) if planes.shape[0] > 1 else planes[0]
    _coeffs, smooth = pdenoise.starlet(L, 3)
    top = L - smooth
    h, w = L.shape
    sample = top[:: max(1, h // 512), :: max(1, w // 512)]
    return 1.4826 * float(np.median(np.abs(sample - np.median(sample))))


class AIDenoise(Tool):
    kind = "ai_denoise"
    title = T("AI 降噪", "AI Noise Reduction")
    help = T("用你自己的資料訓練的 AI 模型（Noise2Noise）：去掉噪聲顆粒和大顆的斑駁，星點與細節的形狀保留得比一般降噪好。"
             "在拉伸前做效果最好。",
             "An AI model trained on your own data (Noise2Noise). It removes grain and blotchy mottle while keeping "
             "stars and detail sharper than conventional noise reduction. Best before stretching.")
    phase = "linear"
    icon = "denoise"

    def defaults(self) -> dict:
        # 去斑、色彩降噪預設關（2026-10-19 使用者試過，覺得沒有它們的第一版效果最好）
        return {"amount": 0.85, "mottle": False, "chroma": False}

    def controls(self) -> list[dict]:
        return [slider("amount", T("強度", "Amount"), 0, 1, 0.05, "{:.0%}",
                       help=T("100%＝完全用 AI 的結果；保留一點原圖會有比較自然的顆粒。",
                              "100% uses the AI result fully; keeping a little of the original looks more natural.")),
                toggle("mottle", T("也去掉大顆的斑駁", "Also remove mottle"),
                       help=T("縮小一半再跑一次 AI，去掉一塊塊的大顆噪聲。只在完整解析度做（快速預覽本來就看不到）。",
                              "Runs the AI again at half size to remove blotchy large-scale noise. Full resolution "
                              "only (the quick preview doesn't show it anyway).")),
                toggle("chroma", T("色彩降噪", "Colour noise reduction"),
                       help=T("背景的彩色小點（綠點、紅點）抹平，亮度細節不動；星點與亮的地方保留顏色。只在完整解析度做。",
                              "Smooths coloured specks in the background without touching luminance detail; stars and "
                              "bright areas keep their colour. Full resolution only."))]

    def run(self, state, params, layer, mask, ctx):
        if not available():
            out = state.copy()
            out.warnings.append(T("AI 降噪的模型還沒準備好（還在訓練或沒有安裝）", "The AI denoise model isn't available yet"))
            return out
        out = super().run(state, params, layer, mask, ctx)
        record_denoise_ratio(state, out, self.target_layers(state, layer), ctx)
        # 記下噪聲（8 px 尺度）降了幾倍：去星量星暈的門檻會乘回去
        for name in self.target_layers(state, layer):
            if state.phase.get(name, "linear") != "linear":
                continue
            before = ops.planes(state.layers[name])
            after = ops.planes(out.layers[name])
            nb = ops.measure_noise(before[0] if before.shape[0] == 1 else before, ctx.scale)
            na = ops.measure_noise(after[0] if after.shape[0] == 1 else after, ctx.scale)
            rl = nb.sigma / max(na.sigma, 1e-12)
            rc = [b / max(a, 1e-12) for b, a in zip(nb.channel_sigma, na.channel_sigma)]
            old = out.notes.get(name, {}).get("noise_ratio")
            if old is not None:
                rl, rc = rl * old[0], [r * o for r, o in zip(rc, old[1])]
            out.notes.setdefault(name, {})["noise_ratio"] = (float(rl), [float(r) for r in rc])
            # 找星用的細節尺度（8 px 以下）噪聲降了幾倍：去星找星時門檻乘回去，不然噪聲變小後
            # 星系的塵埃帶、一團團的雲氣也被當成星點補掉（M31 上留下大片不規則的補痕）
            rs = detail_noise(before) / max(detail_noise(after), 1e-12)
            old = out.notes[name].get("detail_noise_ratio")
            out.notes[name]["detail_noise_ratio"] = float(rs * (old or 1.0))
            ctx.view["noise_ratio"] = round(float(rl), 2)
        return out

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        amount = float(params.get("amount", 0.85))
        if amount <= 0:
            return img
        whole = ctx.whole(img)
        units = []
        for c in range(img.shape[0]):
            sky, sig = sky_sigma(img[c])
            if whole is not img:     # 預覽區：天空從整張量；噪聲是完整解析度的，用這一塊自己量
                sky = float(np.median(whole[c][:: 2, :: 2]))
            units.append((sky, sig))
        # 縮小再跑的那層：快速預覽本身已經縮小（一個像素＝完整尺寸 4 px），斑駁在快速預覽上就是細顆粒，不用再跑
        # 色彩降噪也只在完整解析度做（快速預覽縮小時色彩噪聲已經平均掉了）
        full = ctx.scale < 1.5
        coarse = COARSE if full and params.get("mottle", False) else ()
        chroma = CHROMA_SIGMA / ctx.scale if full and params.get("chroma", False) else 0.0
        _fixed, outs, n_dark = denoise_planes([(img[c] - sky) / sig for c, (sky, sig) in enumerate(units)],
                                              amount, ctx, coarse=coarse, chroma=chroma)
        ctx.view["dark_fixed"] = n_dark
        return np.stack([o * sig + sky for o, (sky, sig) in zip(outs, units)]).astype(np.float32)


register(AIDenoise())

__all__ = ["AIDenoise", "available", "ops"]
