"""訓練 AI 降噪（Noise2Noise）：輸入一張有噪聲的疊圖（A_k），目標是同一片天空另一半 sub 的疊圖（B_k）。

- 單色道模型：彩色影像一個色版一個色版做，單色、彩色、不同相機共用
- 先換成「比天空亮幾倍噪聲」的單位（扣天空、除以噪聲 σ），再用 asinh 壓縮當輸入；網路預測要扣掉的噪聲（殘差）
- 亮處的噪聲比較大（光子噪聲）：用 s＝√(1＋亮度/NOISE_K) 當局部噪聲的尺度。網路的修正量限制在 ±LIMIT·s 以內
  （第一版沒有限制，亮的星系、星雲核心會冒出大片色塊），損失也除以 s²（每個地方都用自己的噪聲衡量）
- 取樣：一部分裁切刻意挑亮的地方（星系、星雲核心、亮星），不然幾乎都是天空，亮處學不到
- 修正量只有細的部分：網路的輸出先減掉自己模糊 HP_SIGMA px 的版本再限制大小（推論時的保護直接做進模型）。
  第三版只在推論時做，網路想整片調亮的地方修正量頂到上限、細的部分被截掉，那些地方完全沒降噪（留下一塊塊彩色顆粒）
- 噪聲質感：自己疊的訓練資料噪聲比較「粗」（對齊插值讓相鄰像素相關），WBPP 的 master 比較「細」；
  一半的樣本在 A、B 各自加上獨立的白噪聲（仍然是合法的 Noise2Noise 配對），模型才不會只認得一種噪聲
  （第二版在 NGC2244 的 WBPP master 上會憑空長出斑塊）
- 第二種資料（make_synth.py）：56 個 master 的小塊當底，訓練時加兩份獨立的模擬噪聲（強度、光子噪聲比例、
  相鄰像素相關程度都隨機），補單色窄帶與星系；占一半樣本。正規化照推論的做法（相鄰差估 σ），所以相關的噪聲
  也會像實際一樣被「低估」
- 噪聲尺度：推論時 σ 是估的，彩色相機的紅色版（去馬賽克後相鄰像素高度相關）會估得太小，換算後噪聲看起來
  比 1 大；所有樣本隨機縮放 0.5～2 倍，模型自己從影像判斷噪聲多大（第五版在 NGC2244 紅色版修正量一直頂到上限）
- 輸入兩個通道：asinh 壓縮的亮度，和「跟周圍（模糊 1.5 px）差多少、以局部噪聲 s 為單位」再 asinh。
  只有第一個通道時，亮處被壓得很扁，±4σ 的噪聲在 300σ 的地方幾乎看不出來，模型分不清噪聲和結構，
  星點、亮星雲的細節被改掉（第六版模擬驗證組：天空 1.29→0.39，但 300σ 以上的誤差變成噪聲的 3 倍）
- 實拍配對的兩半 sub 星點大小、位置有細微差異（seeing、對齊誤差），模型會學到把暗星抹平（第八版暗星的光度
  只剩 73～87%）：實拍配對在亮處（>30σ）和所有星點周圍都不算損失，星點交給模擬配對教（兩份的星點完全一樣）
- 斑駁（--mottle，第九版）：WBPP master 有 4～16 px 的低頻噪聲，模擬噪聲原本只有細顆粒和相鄰像素的相關，
  模型從沒學過，加上修正量高通 3 px，斑駁只減 15%（第八版試過高通 6 px 也沒用，因為資料裡沒有斑駁）。
  模擬噪聲再疊一層低頻噪聲（模糊 2～6 px、強度 0～mottle 倍），兩份各自獨立；驗證也加一項斑駁
- 驗證：M81（沒參加訓練）的 A_4／A_8／A_16 降噪後跟 truth（其餘 sub 的平均，幾乎沒噪聲）比 RMSE；
  另外 NGC6888 Ha（單色）、NGC2244（彩色）的 master 加固定的模擬噪聲，跟 master 本身比

.venv-ai\\Scripts\\python tools/ai/train_n2n.py --data output/ai/n2n --out output/ai/model
（6000 步、每 1000 步驗證一次、存驗證最好的那一版；訓練久了亮處反而變差：訓練資料都是星雲，M81 星系的亮處會被改動）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 192
SOFT = 3.0          # asinh 壓縮的尺度（σ 單位）
NOISE_K = 16.0      # 局部噪聲尺度 s＝√(1＋亮度/NOISE_K)
LIMIT = 6.0         # 修正量上限（s 的倍數）
BRIGHT_SHARE = 0.4  # 挑亮處裁切的比例
WHITE_SHARE = 0.5   # 加白噪聲的樣本比例
HP_SIGMA = 3.0      # 修正量去掉比這個模糊尺度大的部分
SYNTH_SHARE = 0.5   # 模擬噪聲樣本的比例
SCALE_RANGE = (0.8, 1.25)  # 噪聲尺度隨機縮放的範圍（第七版用 0.5～2：有結構的地方模型猜不準噪聲多大）
IN_SIGMA = 1.5      # 第二個輸入通道：跟模糊這麼多的版本比
REAL_BRIGHT = 30.0  # 實拍配對比這個亮的地方不算損失
REAL_STAR = 2.5     # 實拍配對：細節（模糊 1 px 減模糊 4 px）超過這麼多倍噪聲的地方是星點，連周圍都不算損失
F16_MAX = 6.0e4


# ---------------------------------------------------------------------- 資料


def robust_sky_sigma(a: np.ndarray, dist: int | None = None) -> tuple[float, float]:
    """天空（中位數）與噪聲（相隔 dist 列的像素差的 MAD），抽樣算。
    dist＝1（第八版以前）：相鄰像素差。去馬賽克、對齊插值讓相鄰像素相關，σ 會量得太小，而且太小多少看疊圖軟體：
    Photons（雙線性去馬賽克）的紅、藍色版小 1.3～2 倍、WBPP 大多 1～1.5 倍 → 兩種配對的噪聲單位不一樣，混在一起訓練的
    模型（第十版）在亮背景每顆星外面一圈環（輸入噪聲放大 1.5 倍環就消失）。dist＝4（第十三版）：隔 4 px 已經不相關，
    量到的是真正每個像素的噪聲。dist 用環境變數 N2N_NOISE_DIST 傳（Windows 的資料載入子行程不會帶到主程式改的全域變數），
    推論端照模型旁邊的 n2n.json 用一樣的距離（studio/tools/ai.py 的 sky_sigma）"""
    dist = int(os.environ.get("N2N_NOISE_DIST", "1")) if dist is None else dist
    s = a[:: 4, :: 4].astype(np.float32)
    sky = float(np.median(s))
    if dist == 1:
        d = (a[1::2, ::2].astype(np.float32) - a[:-1:2, ::2].astype(np.float32))[::2, ::2]
    else:
        d = (a[dist:, ::2].astype(np.float32) - a[:-dist, ::2].astype(np.float32))[::4, ::2]
    sigma = float(np.median(np.abs(d - np.median(d)))) * 1.4826 / math.sqrt(2)
    return sky, max(sigma, 1e-7)


def correlated_noise(shape: tuple[int, int], rng: np.random.Generator, sigma_c: float) -> np.ndarray:
    """標準差 1 的噪聲；sigma_c > 0 時用高斯模糊做出相鄰像素的相關（像去馬賽克、對齊插值）。"""
    n = rng.standard_normal(shape).astype(np.float32)
    if sigma_c > 0.05:
        from scipy import ndimage
        n = ndimage.gaussian_filter(n, sigma_c, mode="reflect")
        n /= max(float(n.std()), 1e-6)
    return n


def mottle_noise(shape: tuple[int, int], rng: np.random.Generator, size: float) -> np.ndarray:
    """標準差 1 的低頻噪聲（模糊 size px）：疊圖的斑駁。"""
    from scipy import ndimage
    n = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), size, mode="reflect")
    return n / max(float(n.std()), 1e-6)


def synth_pair(m: np.ndarray, rng: np.random.Generator, g: float, S: float,
               sigma_c: float, mottle: float = 0.0, mottle_size: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """master 的一塊（master 的 σ 單位）→ (A, B, 答案)，都已經照推論的方式正規化。
    g：加的噪聲是 master 噪聲的幾倍；S：天空在噪聲單位的亮度（光子噪聲 var＝1＋亮度/S）；
    mottle：再疊多少（細顆粒的幾倍）模糊 mottle_size px 的低頻噪聲。"""
    clean = m.astype(np.float32) / g
    std = np.sqrt(1 + np.maximum(clean, 0) / S)
    n1 = correlated_noise(m.shape, rng, sigma_c)
    n2 = correlated_noise(m.shape, rng, sigma_c)
    if mottle > 0:
        n1 = n1 + mottle * mottle_noise(m.shape, rng, mottle_size)
        n2 = n2 + mottle * mottle_noise(m.shape, rng, mottle_size)
    est = robust_sky_sigma(n1)[1]                # 推論時在天空量到的 σ（相關的噪聲會偏小）
    return (clean + std * n1) / est, (clean + std * n2) / est, clean / est


class Synth:
    """master 小塊＋即時加的模擬噪聲。"""

    def __init__(self, root: Path, seed: int = 1, mottle: float = 0.0):
        self.mottle = mottle               # 存在物件上：Windows 的資料載入子行程不會帶到主程式改的全域變數
        self.crops = []
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        for m in meta:
            if not m["holdout"]:
                self.crops.append(np.load(root / f"{m['name']}.npy", mmap_mode="r"))
        self.sizes = np.array([len(c) for c in self.crops], np.float64)
        self.seed = seed

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        arr = self.crops[rng.choice(len(self.crops), p=self.sizes / self.sizes.sum())]
        m = arr[int(rng.integers(len(arr)))]
        y0, x0 = (int(rng.integers(0, m.shape[0] - CROP + 1)), int(rng.integers(0, m.shape[1] - CROP + 1)))
        m = np.asarray(m[y0:y0 + CROP, x0:x0 + CROP], np.float32)
        m = np.nan_to_num(np.clip(m, -F16_MAX, F16_MAX), nan=0.0)     # 很深的 master 星核會超出 float16
        g = float(np.exp(rng.uniform(np.log(2.0), np.log(10.0))))
        S = float(np.exp(rng.uniform(np.log(8.0), np.log(400.0))))
        sigma_c = float(rng.uniform(0.3, 1.1)) if rng.random() < 0.7 else 0.0
        mt = float(rng.uniform(0.0, self.mottle)) if self.mottle > 0 and rng.random() < 0.7 else 0.0
        a, b, _t = synth_pair(m, rng, g, S, sigma_c, mt, float(rng.uniform(2.0, 6.0)))
        return a, b


class Pairs(torch.utils.data.IterableDataset):
    def __init__(self, root: Path, train: bool, seed: int = 0):
        self.items = []
        for meta_path in sorted(root.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("holdout") == train:
                continue
            for k in meta["levels"]:
                self.items.append((meta_path.parent, k, meta["shape"]))
        self.seed = seed
        self.stats: dict = {}
        self.bright: dict = {}
        self.synth: Synth | None = None
        self.scale_range = SCALE_RANGE     # 存在物件上：Windows 的資料載入子行程不會帶到主程式改的全域變數

    def _arr(self, folder: Path, name: str) -> np.ndarray:
        return np.load(folder / f"{name}.npy", mmap_mode="r")

    def _stats(self, folder: Path, k: int, c: int, A: np.ndarray) -> tuple[float, float]:
        key = (str(folder), k, c)
        if key not in self.stats:
            self.stats[key] = robust_sky_sigma(np.asarray(A[c]))
        return self.stats[key]

    def _bright(self, folder: Path, k: int, A: np.ndarray, sky: float, sig: float) -> np.ndarray:
        """亮的地方（比天空亮 15σ 以上）的座標，8 px 一格抽樣。"""
        key = (str(folder), k)
        if key not in self.bright:
            L = np.asarray(A[:, ::8, ::8], np.float32).mean(axis=0)
            ys, xs = np.nonzero(L > sky + 15 * sig)
            self.bright[key] = np.column_stack([ys * 8, xs * 8]) if len(ys) else np.zeros((0, 2), int)
        return self.bright[key]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 9973 + int(time.time()))
        weights = np.array([s[1] * s[2] for _f, _k, s in self.items], np.float64)
        weights /= weights.sum()
        while True:
            if self.synth is not None and rng.random() < SYNTH_SHARE:
                a, b = self.synth.sample(rng)
                yield (*self._finish(a, b, rng), torch.tensor(1.0))
                continue
            folder, k, shape = self.items[rng.choice(len(self.items), p=weights)]
            A, B = self._arr(folder, f"A_{k}"), self._arr(folder, f"B_{k}")
            c = int(rng.integers(shape[0]))
            sky, sig = self._stats(folder, k, c, A)
            y0 = int(rng.integers(0, shape[1] - CROP))
            x0 = int(rng.integers(0, shape[2] - CROP))
            if rng.random() < BRIGHT_SHARE:
                pts = self._bright(folder, k, A, sky, sig)
                if len(pts):
                    cy, cx = pts[int(rng.integers(len(pts)))]
                    y0 = int(np.clip(cy - CROP // 2 + rng.integers(-48, 49), 0, shape[1] - CROP))
                    x0 = int(np.clip(cx - CROP // 2 + rng.integers(-48, 49), 0, shape[2] - CROP))
            a = (np.asarray(A[c, y0:y0 + CROP, x0:x0 + CROP], np.float32) - sky) / sig
            b = (np.asarray(B[c, y0:y0 + CROP, x0:x0 + CROP], np.float32) - sky) / sig
            if rng.random() < WHITE_SHARE:
                u = float(rng.uniform(0.3, 1.5))
                k = 1.0 / math.sqrt(1 + u * u)
                a = (a + u * rng.standard_normal(a.shape).astype(np.float32)) * k
                b = (b + u * rng.standard_normal(b.shape).astype(np.float32)) * k
            yield (*self._finish(a, b, rng), torch.tensor(0.0))

    def _finish(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
        """噪聲尺度隨機縮放、對調、旋轉、翻轉。"""
        lo, hi = self.scale_range
        u = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        a, b = a * u, b * u
        if rng.random() < 0.5:
            a, b = b, a
        r = int(rng.integers(4))
        a, b = np.rot90(a, r), np.rot90(b, r)
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(a, np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(b, np.float32))[None])


# ---------------------------------------------------------------------- 網路


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))


def noise_scale(x: torch.Tensor) -> torch.Tensor:
    """局部噪聲的尺度（σ 單位）：天空是 1，亮處照光子噪聲變大。"""
    level = F.avg_pool2d(torch.clamp(x, min=0), 5, 1, 2)
    return torch.sqrt(1 + level / NOISE_K)


def gaussian_kernel(sigma: float) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


class UNet(nn.Module):
    """輸入 asinh 壓縮過的影像（σ 單位），輸出要扣掉的噪聲（σ 單位、線性），限制在 ±LIMIT·s。"""

    def __init__(self, ch: tuple[int, ...] = (32, 64, 128, 192), hp_sigma: float = HP_SIGMA):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 2
        for c in ch:
            self.enc.append(block(cin, c))
            cin = c
        self.dec = nn.ModuleList()
        for c_skip, c in zip(reversed(ch[:-1]), reversed(ch[1:])):
            self.dec.append(block(c + c_skip, c_skip))
        self.head = nn.Conv2d(ch[0], 1, 1)
        k = gaussian_kernel(hp_sigma)
        self.register_buffer("gk_h", k.view(1, 1, 1, -1))
        self.register_buffer("gk_v", k.view(1, 1, -1, 1))
        k = gaussian_kernel(IN_SIGMA)
        self.register_buffer("gi_h", k.view(1, 1, 1, -1))
        self.register_buffer("gi_v", k.view(1, 1, -1, 1))

    @staticmethod
    def _blur(r: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        n = kh.shape[-1] // 2
        b = F.conv2d(F.pad(r, (n, n, 0, 0), mode="reflect"), kh)
        return F.conv2d(F.pad(b, (0, 0, n, n), mode="reflect"), kv)

    def highpass(self, r: torch.Tensor) -> torch.Tensor:
        return r - self._blur(r, self.gk_h, self.gk_v)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(亮度 asinh, 跟周圍的差／局部噪聲 再 asinh)；float32 算。"""
        def asinh(v: torch.Tensor) -> torch.Tensor:
            return torch.sign(v) * torch.log(torch.abs(v) + torch.sqrt(v * v + 1))   # 這樣寫才能匯出 ONNX
        detail = (x - self._blur(x, self.gi_h, self.gi_v)) / noise_scale(x)
        return torch.cat([asinh(x / SOFT), asinh(detail / 2)], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.features(x.float())
        skips = []
        for i, e in enumerate(self.enc):
            z = e(z)
            if i < len(self.enc) - 1:
                skips.append(z)
                z = F.avg_pool2d(z, 2)
        for d in self.dec:
            s = skips.pop()
            z = F.interpolate(z, size=s.shape[-2:], mode="bilinear", align_corners=False)
            z = d(torch.cat([z, s], 1))
        r = self.head(z).float()
        # 高通與限制在 float32 做（混合精度訓練時 fp16 的卷積會溢位變 NaN）
        with torch.autocast(device_type=x.device.type, enabled=False):
            r = self.highpass(r)
            residual = noise_scale(x.float()) * LIMIT * torch.tanh(r / LIMIT)
            return x.float() - residual


def loss_fn(pred: torch.Tensor, target: torch.Tensor, inp: torch.Tensor, synth: torch.Tensor,
            l1: bool = False, l2_mix: float = 0.0) -> torch.Tensor:
    # 每個地方用自己的噪聲衡量：除以 s²（L1 除以 s）；權重只看輸入（不影響 Noise2Noise 的無偏性）。
    # 實拍配對的亮處不算（兩半 sub 的星點本來就不一樣）
    # l1（第十一版）：平方誤差學到的是「平均」，孤立的一個亮點（可能是噪聲、少數時候是暗星）會留一部分，
    # 背景變平之後就是一顆顆單點；絕對誤差學到的是「中位數」，比較可能是噪聲就整個拿掉，
    # 真的暗星有好幾個像素一起亮、比較可能是星，就整個留著。
    # 但純 L1 不在乎少數像素錯很多：亮背景（IC434 左邊 15～25σ）每顆暗星外面一圈環（修正量頂到上限 0.26%）。
    # l2_mix：再加 l2_mix × 平方誤差，小誤差照 L1（去單點），大誤差照平方誤差（不准錯很多）
    with torch.no_grad():
        w = 1.0 / noise_scale(inp) ** 2
        level = F.avg_pool2d(torch.clamp(inp, min=0), 5, 1, 2)
        real = (synth < 0.5).view(-1, 1, 1, 1)
        both = (inp + target) / 2                                 # 兩份平均，噪聲小一點，找星比較準
        detail = F.avg_pool2d(both, 3, 1, 1) - F.avg_pool2d(both, 9, 1, 4)
        star = F.max_pool2d((detail > REAL_STAR * 0.6).float(), 9, 1, 4) > 0      # 星點，外擴 4 px
        w = torch.where(real & ((level > REAL_BRIGHT) | star), torch.zeros_like(w), w)
    if l1:
        e = pred - target
        return (w.sqrt() * e.abs() + l2_mix * w * e * e).sum() / w.numel()
    return (w * (pred - target) ** 2).sum() / w.numel()


# ---------------------------------------------------------------------- 驗證


def tiled(model: nn.Module, x: np.ndarray, device, tile: int = 512, pad: int = 32) -> np.ndarray:
    h, w = x.shape
    out = np.empty_like(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for y in range(0, h, tile):
            for xx in range(0, w, tile):
                y0, x0 = max(0, y - pad), max(0, xx - pad)
                y1, x1 = min(h, y + tile + pad), min(w, xx + tile + pad)
                t = torch.from_numpy(np.ascontiguousarray(x[y0:y1, x0:x1]))[None, None].to(device)
                p = model(t).float()[0, 0].cpu().numpy()
                out[y:min(h, y + tile), xx:min(w, xx + tile)] = p[y - y0:y - y0 + min(tile, h - y),
                                                                  xx - x0:xx - x0 + min(tile, w - xx)]
    return out


def validate(model: nn.Module, root: Path, device) -> dict:
    model.eval()
    res = {}
    for meta_path in root.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("holdout"):
            continue
        folder = meta_path.parent
        truth = np.load(folder / "truth.npy", mmap_mode="r")
        h, w = truth.shape[1:]
        box = (slice(h // 2 - 600, h // 2 + 600), slice(w // 2 - 900, w // 2 + 900))
        for k in meta["levels"][:3]:
            A = np.load(folder / f"A_{k}.npy", mmap_mode="r")
            e = {"raw_sky": [], "ai_sky": [], "raw_bright": [], "ai_bright": []}
            for c in range(A.shape[0]):
                sky, sig = robust_sky_sigma(np.asarray(A[c]))
                a = (np.asarray(A[c][box], np.float32) - sky) / sig
                t = (np.asarray(truth[c][box], np.float32) - sky) / sig
                p = tiled(model, a, device)
                # 亮處用局部噪聲正規化（光子噪聲），不然被亮星主導
                sc = np.sqrt(1 + np.maximum(t, 0) / NOISE_K)
                skym, brightm = t < 3, t > 15
                for name, img in (("raw", a), ("ai", p)):
                    d = (img - t) / sc
                    e[f"{name}_sky"].append(float(np.sqrt(np.mean(d[skym] ** 2))))
                    e[f"{name}_bright"].append(float(np.sqrt(np.mean(d[brightm] ** 2))) if brightm.any() else 0.0)
            res[f"{folder.name}_k{k}"] = {n: round(float(np.mean(v)), 3) for n, v in e.items()}
    model.train()
    return res


def validate_synth(model: nn.Module, root: Path, device) -> dict:
    """驗證組 master 加固定的模擬噪聲（一半白噪聲、一半相關噪聲），跟 master 本身比。"""
    model.eval()
    res = {}
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    for m in meta:
        if not m["holdout"]:
            continue
        crops = np.load(root / f"{m['name']}.npy")
        rng = np.random.default_rng(123)
        crops = np.nan_to_num(np.clip(crops.astype(np.float32), -F16_MAX, F16_MAX), nan=0.0)
        e = {f"{w}_{k}": [] for k in ("sky", "faint", "bright", "core", "mottle") for w in ("raw", "ai")}
        from scipy import ndimage
        for i, crop in enumerate(crops):
            a, _b, t = synth_pair(crop, rng, 4.0, 50.0, 0.7 if i % 2 else 0.0)
            p = tiled(model, a.astype(np.float32), device)
            sc = np.sqrt(1 + np.maximum(t, 0) / 50.0)      # 實際加的噪聲（光子噪聲 S＝50）
            for name, img in (("raw", a), ("ai", p)):
                d = ((img - t) / sc) ** 2
                for key, sel in (("sky", t < 3), ("faint", (t >= 3) & (t <= 15)), ("bright", (t > 15) & (t < 300)),
                                 ("core", t >= 300)):
                    if sel.any():
                        e[f"{name}_{key}"].append(d[sel])
            # 斑駁：同一塊加有斑駁的噪聲（0.4 倍、模糊 4 px），量誤差模糊 4 px 後還剩多少（天空）
            a, _b, t = synth_pair(crop, rng, 4.0, 50.0, 0.7 if i % 2 else 0.0, 0.4, 4.0)
            p = tiled(model, a.astype(np.float32), device)
            for name, img in (("raw", a), ("ai", p)):
                d = ndimage.gaussian_filter(img - t, 4.0) ** 2
                e[f"{name}_mottle"].append(d[t < 3])
        res[m["name"]] = {n: round(float(np.sqrt(np.mean(np.concatenate(v)))) if v else 0.0, 3) for n, v in e.items()}
    model.train()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/ai/n2n")
    ap.add_argument("--out", default="output/ai/model")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--synth", default="output/ai/synth")
    ap.add_argument("--scale", type=float, nargs=2, default=list(SCALE_RANGE), help="噪聲尺度隨機縮放的範圍")
    ap.add_argument("--width", type=int, nargs=4, default=[48, 96, 192, 256], help="U-Net 各層的通道數")
    ap.add_argument("--hp", type=float, default=HP_SIGMA, help="修正量只留比這個模糊尺度細的部分（px）")
    ap.add_argument("--mottle", type=float, default=0.0, help="模擬噪聲再疊多少倍的斑駁（低頻噪聲，第九版）")
    ap.add_argument("--noise-dist", type=int, default=1,
                    help="量噪聲用相隔幾列的像素差（1＝相鄰，第八版；4＝不受去馬賽克影響，第十三版）")
    ap.add_argument("--loss", choices=("l2", "l1", "l1l2"), default="l2",
                    help="l1：學中位數（孤立單點整個拿掉，第十一版）；l1l2：L1＋0.25×平方誤差（大誤差不准，避免星點外的環）")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = UNet(tuple(a.width), hp_sigma=a.hp).to(device)
    (out / "config.json").write_text(json.dumps({"width": a.width, "scale": a.scale, "hp": a.hp, "mottle": a.mottle, "loss": a.loss,
                                                "noise_dist": a.noise_dist}),
                                     encoding="utf-8")
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    os.environ["N2N_NOISE_DIST"] = str(a.noise_dist)
    pairs = Pairs(root, train=True)
    pairs.scale_range = tuple(a.scale)
    synth_root = Path(a.synth) if a.synth else None
    if synth_root is not None and (synth_root / "meta.json").is_file():
        pairs.synth = Synth(synth_root, mottle=a.mottle)
        print("synthetic crops", int(pairs.synth.sizes.sum()), flush=True)
    loader = torch.utils.data.DataLoader(pairs, batch_size=a.batch, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    t0 = time.time()
    log = []
    run = 0.0
    best = None          # (分數, step)：驗證組天空＋亮處的誤差和，存最好的那一版
    skipped = 0
    for step, (x, y, kind) in enumerate(loader, start=1):
        x, y, kind = x.to(device, non_blocking=True), y.to(device, non_blocking=True), kind.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(x)
        loss = loss_fn(pred.float(), y, x, kind, l1=a.loss in ("l1", "l1l2"), l2_mix=0.25 if a.loss == "l1l2" else 0.0)
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss.detach()) if step > 1 else float(loss.detach())
        if step % 200 == 0:
            print(f"step {step} loss {run:.4f} {(time.time() - t0) / step * 1000:.0f} ms/step skipped {skipped}",
                  flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, root, device)
            if pairs.synth is not None:
                v.update(validate_synth(model, synth_root, device))
            log.append({"step": step, "loss": run, "val": v})
            print("  val (RMSE in local σ):", v, flush=True)
            # 每一項都跟原圖比（AI／原圖），加起來；亮處、星核不能比原圖差太多
            score = sum(d[f"ai_{k}"] / d[f"raw_{k}"] for d in v.values()
                        for k in ("sky", "faint", "bright", "core", "mottle") if d.get(f"raw_{k}", 0) > 0)
            if math.isfinite(score) and (best is None or score < best[0]):
                best = (score, step)
                torch.save(model.state_dict(), out / "n2n.pt")
                print(f"  best so far (score {score:.3f})", flush=True)
            (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if step >= a.steps:
            break
    model.load_state_dict(torch.load(out / "n2n.pt"))
    print("export best step", best[1] if best else None, flush=True)
    model.eval().float()
    dummy = torch.zeros(1, 1, 256, 256, device=device)
    torch.onnx.export(model, dummy, out / "n2n.onnx", input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}, dynamo=False)
    # 推論端要用一樣的方法量噪聲（studio/tools/ai.py 讀模型旁邊的 n2n.json；沒有這個檔＝相鄰像素差）
    (out / "n2n.json").write_text(json.dumps({"noise_dist": a.noise_dist}), encoding="utf-8")
    print("exported", out / "n2n.onnx", f"total {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
