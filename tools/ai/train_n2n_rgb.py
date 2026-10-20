"""訓練彩色 AI 降噪（三個色版一起看）：Noise2Noise，做法跟 train_n2n.py（單色道）一樣，差別在輸入輸出都是三色。

為什麼：單色道模型一個色版一個色版做，三色彼此不知道對方，剩下的噪聲各色不一樣 → 背景一顆顆綠點、紅點，
星點的顏色也偏一點（模擬：星色變差的星 6%）。三色一起看，模型分得出「三色一起亮的是結構、只有一色的是噪聲」。
單色影像繼續用單色道模型（studio/tools/ai.py 依色版數選）。

- 輸入：每個色版各自換成噪聲 σ 單位（扣天空、除以相鄰差估的 σ，跟推論一樣），每色兩個特徵（asinh 亮度、
  跟周圍差多少／局部噪聲），共 6 個通道；輸出三色要扣掉的噪聲，每色各自高通、限制 ±LIMIT·s（s 用自己色版的亮度）
- 資料：實拍配對只用彩色的（7 組）；模擬配對用 make_synth.py --rgb 的三色小塊，三色各自加噪聲，
  色版之間有隨機的相關（0～RHO_MAX，去馬賽克讓相鄰色版的噪聲相關）。實拍 OSC master 天空的色版間相關 0.4～0.8
  （LBN576 0.63、M31 0.73～0.83、IC434 0.51～0.81）；第一版只到 0.6，三色一起暗／亮的噪聲被當成結構留下，
  背景一堆白、藍、黑的單點（LBN576 >0.8σ 的暗點是單色道模型的 5 倍）
- 損失、亮處取樣、實拍配對星點不算、存最好的那一版：跟單色道一樣；驗證多一項色彩誤差（各色減三色平均）

.venv-ai\\Scripts\\python tools/ai/train_n2n_rgb.py --data output/ai/n2n --synth output/ai/synth_rgb --out output/ai/model_rgb1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_n2n import (BRIGHT_SHARE, CROP, F16_MAX, HP_SIGMA, IN_SIGMA, LIMIT, NOISE_K, REAL_BRIGHT,  # noqa: E402
                       REAL_STAR, SCALE_RANGE, SOFT, SYNTH_SHARE, WHITE_SHARE, block, correlated_noise,
                       gaussian_kernel, robust_sky_sigma)

C = 3
# 網路輸出被高通拿掉的低頻部分不影響結果，訓練時沒有東西拉住它：第一次訓練一路漂到幾千（單色道第八版只有 30），
# fp16 溢位、跳過的步數越來越多、驗證出現 NaN。加一點懲罰把它拉回 0（跟輸出無關，不影響降噪）
LOW_WEIGHT = 1e-2
RHO_MAX = 0.9       # 色版之間噪聲相關的上限（見最上面）


# ---------------------------------------------------------------------- 資料


def synth_pair3(m: np.ndarray, rng: np.random.Generator, g: float, S: float, sigma_c: float,
                rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三色小塊（各色 master 的 σ 單位）→ (A, B, 答案)，每色照推論的方式正規化。
    rho：色版之間噪聲的相關係數（三色共用一份噪聲的比例）。"""
    clean = m.astype(np.float32) / g
    std = np.sqrt(1 + np.maximum(clean, 0) / S)

    def noise() -> np.ndarray:
        shared = correlated_noise(m.shape[1:], rng, sigma_c)
        return np.stack([math.sqrt(1 - rho) * correlated_noise(m.shape[1:], rng, sigma_c) + math.sqrt(rho) * shared
                         for _ in range(m.shape[0])])
    n1, n2 = noise(), noise()
    est = np.array([robust_sky_sigma(n1[c])[1] for c in range(m.shape[0])], np.float32)[:, None, None]
    return (clean + std * n1) / est, (clean + std * n2) / est, clean / est


class Synth3:
    def __init__(self, root: Path):
        self.crops = []
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        for m in meta:
            if not m["holdout"]:
                self.crops.append(np.load(root / f"{m['name']}.npy", mmap_mode="r"))
        self.sizes = np.array([len(c) for c in self.crops], np.float64)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        arr = self.crops[rng.choice(len(self.crops), p=self.sizes / self.sizes.sum())]
        m = arr[int(rng.integers(len(arr)))]
        y0, x0 = int(rng.integers(0, m.shape[1] - CROP + 1)), int(rng.integers(0, m.shape[2] - CROP + 1))
        m = np.asarray(m[:, y0:y0 + CROP, x0:x0 + CROP], np.float32)
        m = np.nan_to_num(np.clip(m, -F16_MAX, F16_MAX), nan=0.0)
        g = float(np.exp(rng.uniform(np.log(2.0), np.log(10.0))))
        S = float(np.exp(rng.uniform(np.log(8.0), np.log(400.0))))
        sigma_c = float(rng.uniform(0.3, 1.1)) if rng.random() < 0.7 else 0.0
        rho = float(rng.uniform(0.0, RHO_MAX))
        a, b, _t = synth_pair3(m, rng, g, S, sigma_c, rho)
        return a, b


class Pairs3(torch.utils.data.IterableDataset):
    def __init__(self, root: Path, train: bool, seed: int = 0):
        self.items = []
        for meta_path in sorted(root.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("holdout") == train or meta["shape"][0] != C:
                continue
            for k in meta["levels"]:
                self.items.append((meta_path.parent, k, meta["shape"]))
        self.seed = seed
        self.stats: dict = {}
        self.bright: dict = {}
        self.synth: Synth3 | None = None
        self.scale_range = SCALE_RANGE

    def _arr(self, folder: Path, name: str) -> np.ndarray:
        return np.load(folder / f"{name}.npy", mmap_mode="r")

    def _stats(self, folder: Path, k: int, A: np.ndarray) -> list[tuple[float, float]]:
        key = (str(folder), k)
        if key not in self.stats:
            self.stats[key] = [robust_sky_sigma(np.asarray(A[c])) for c in range(C)]
        return self.stats[key]

    def _bright(self, folder: Path, k: int, A: np.ndarray, st: list) -> np.ndarray:
        key = (str(folder), k)
        if key not in self.bright:
            L = np.mean([(np.asarray(A[c, ::8, ::8], np.float32) - s) / g for c, (s, g) in enumerate(st)], axis=0)
            ys, xs = np.nonzero(L > 15)
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
            st = self._stats(folder, k, A)
            y0 = int(rng.integers(0, shape[1] - CROP))
            x0 = int(rng.integers(0, shape[2] - CROP))
            if rng.random() < BRIGHT_SHARE:
                pts = self._bright(folder, k, A, st)
                if len(pts):
                    cy, cx = pts[int(rng.integers(len(pts)))]
                    y0 = int(np.clip(cy - CROP // 2 + rng.integers(-48, 49), 0, shape[1] - CROP))
                    x0 = int(np.clip(cx - CROP // 2 + rng.integers(-48, 49), 0, shape[2] - CROP))
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            a = np.stack([(np.asarray(A[c][sl], np.float32) - s) / g for c, (s, g) in enumerate(st)])
            b = np.stack([(np.asarray(B[c][sl], np.float32) - s) / g for c, (s, g) in enumerate(st)])
            if rng.random() < WHITE_SHARE:
                u = float(rng.uniform(0.3, 1.5))
                kk = 1.0 / math.sqrt(1 + u * u)
                a = (a + u * rng.standard_normal(a.shape).astype(np.float32)) * kk
                b = (b + u * rng.standard_normal(b.shape).astype(np.float32)) * kk
            yield (*self._finish(a, b, rng), torch.tensor(0.0))

    def _finish(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
        """噪聲尺度每色各自隨機縮放（推論時每色的 σ 是各自估的）、對調、旋轉、翻轉。"""
        lo, hi = self.scale_range
        u = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(C, 1, 1))).astype(np.float32)
        a, b = a * u, b * u
        if rng.random() < 0.5:
            a, b = b, a
        r = int(rng.integers(4))
        a, b = np.rot90(a, r, axes=(1, 2)), np.rot90(b, r, axes=(1, 2))
        if rng.random() < 0.5:
            a, b = a[:, :, ::-1], b[:, :, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(a, np.float32)),
                torch.from_numpy(np.ascontiguousarray(b, np.float32)))


# ---------------------------------------------------------------------- 網路


def noise_scale(x: torch.Tensor) -> torch.Tensor:
    """每個色版的局部噪聲尺度（σ 單位）。"""
    level = F.avg_pool2d(torch.clamp(x, min=0), 5, 1, 2)
    return torch.sqrt(1 + level / NOISE_K)


class UNet3(nn.Module):
    """輸入三色（σ 單位），輸出三色降噪後的影像；每色的修正量各自高通、限制在 ±LIMIT·s。"""

    def __init__(self, ch: tuple[int, ...] = (48, 96, 192, 256), hp_sigma: float = HP_SIGMA):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 2 * C
        for c in ch:
            self.enc.append(block(cin, c))
            cin = c
        self.dec = nn.ModuleList()
        for c_skip, c in zip(reversed(ch[:-1]), reversed(ch[1:])):
            self.dec.append(block(c + c_skip, c_skip))
        self.head = nn.Conv2d(ch[0], C, 1)
        k = gaussian_kernel(hp_sigma)
        self.register_buffer("gk_h", k.view(1, 1, 1, -1).repeat(C, 1, 1, 1))
        self.register_buffer("gk_v", k.view(1, 1, -1, 1).repeat(C, 1, 1, 1))
        k = gaussian_kernel(IN_SIGMA)
        self.register_buffer("gi_h", k.view(1, 1, 1, -1).repeat(C, 1, 1, 1))
        self.register_buffer("gi_v", k.view(1, 1, -1, 1).repeat(C, 1, 1, 1))

    @staticmethod
    def _blur(r: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        n = kh.shape[-1] // 2
        b = F.conv2d(F.pad(r, (n, n, 0, 0), mode="reflect"), kh, groups=C)
        return F.conv2d(F.pad(b, (0, 0, n, n), mode="reflect"), kv, groups=C)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        def asinh(v: torch.Tensor) -> torch.Tensor:
            return torch.sign(v) * torch.log(torch.abs(v) + torch.sqrt(v * v + 1))
        detail = (x - self._blur(x, self.gi_h, self.gi_v)) / noise_scale(x)
        return torch.cat([asinh(x / SOFT), asinh(detail / 2)], 1)

    def forward(self, x: torch.Tensor, with_low: bool = False):
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            low = self._blur(r, self.gk_h, self.gk_v)
            r = r - low
            residual = noise_scale(x.float()) * LIMIT * torch.tanh(r / LIMIT)
            pred = x.float() - residual
            return (pred, low) if with_low else pred


def loss_fn(pred: torch.Tensor, target: torch.Tensor, inp: torch.Tensor, synth: torch.Tensor) -> torch.Tensor:
    """跟單色道一樣：每色除以自己的 s²；實拍配對的亮處與星點（用三色平均找）不算。"""
    with torch.no_grad():
        w = 1.0 / noise_scale(inp) ** 2
        L = inp.mean(1, keepdim=True)
        level = F.avg_pool2d(torch.clamp(L, min=0), 5, 1, 2)
        real = (synth < 0.5).view(-1, 1, 1, 1)
        both = (inp + target).mean(1, keepdim=True) / 2
        detail = F.avg_pool2d(both, 3, 1, 1) - F.avg_pool2d(both, 9, 1, 4)
        star = F.max_pool2d((detail > REAL_STAR * 0.6).float(), 9, 1, 4) > 0
        drop = (real & ((level > REAL_BRIGHT) | star)).expand_as(w)
        w = torch.where(drop, torch.zeros_like(w), w)
    return (w * (pred - target) ** 2).sum() / w.numel()


# ---------------------------------------------------------------------- 驗證


def tiled(model: nn.Module, x: np.ndarray, device, tile: int = 512, pad: int = 32) -> np.ndarray:
    _c, h, w = x.shape
    out = np.empty_like(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for y in range(0, h, tile):
            for xx in range(0, w, tile):
                y0, x0 = max(0, y - pad), max(0, xx - pad)
                y1, x1 = min(h, y + tile + pad), min(w, xx + tile + pad)
                t = torch.from_numpy(np.ascontiguousarray(x[:, y0:y1, x0:x1]))[None].to(device)
                p = model(t).float()[0].cpu().numpy()
                out[:, y:min(h, y + tile), xx:min(w, xx + tile)] = p[:, y - y0:y - y0 + min(tile, h - y),
                                                                     xx - x0:xx - x0 + min(tile, w - xx)]
    return out


def _chroma_err(img: np.ndarray, t: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """各色減三色平均的誤差（模糊 2 px：彩色小點、色斑）。"""
    from scipy import ndimage
    e = (img - img.mean(0)) - (t - t.mean(0))
    return np.stack([ndimage.gaussian_filter(v, 2.0) for v in e])[:, sel]


def validate(model: nn.Module, root: Path, device) -> dict:
    model.eval()
    res = {}
    for meta_path in root.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("holdout") or meta["shape"][0] != C:
            continue
        folder = meta_path.parent
        truth = np.load(folder / "truth.npy", mmap_mode="r")
        h, w = truth.shape[1:]
        box = (slice(h // 2 - 600, h // 2 + 600), slice(w // 2 - 900, w // 2 + 900))
        k = meta["levels"][0]                   # 只有 A_4 跟 truth 沒有重疊（truth＝另外 20 張）
        A = np.load(folder / f"A_{k}.npy", mmap_mode="r")
        st = [robust_sky_sigma(np.asarray(A[c])) for c in range(C)]
        a = np.stack([(np.asarray(A[c][box], np.float32) - s) / g for c, (s, g) in enumerate(st)])
        t = np.stack([(np.asarray(truth[c][box], np.float32) - s) / g for c, (s, g) in enumerate(st)])
        p = tiled(model, a, device)
        tl = t.mean(0)
        sc = np.sqrt(1 + np.maximum(tl, 0) / NOISE_K)
        e = {}
        for name, img in (("raw", a), ("ai", p)):
            d = (img.mean(0) - tl) / sc
            e[f"{name}_sky"] = float(np.sqrt(np.mean(d[tl < 3] ** 2)))
            e[f"{name}_bright"] = float(np.sqrt(np.mean(d[tl > 15] ** 2)))
            e[f"{name}_chroma"] = float(np.sqrt(np.mean(_chroma_err(img, t, tl < 3) ** 2)))
        res[f"{folder.name}_k{k}"] = {n: round(v, 3) for n, v in e.items()}
    model.train()
    return res


def validate_synth(model: nn.Module, root: Path, device) -> dict:
    model.eval()
    res = {}
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    for m in meta:
        if not m["holdout"]:
            continue
        crops = np.nan_to_num(np.clip(np.load(root / f"{m['name']}.npy").astype(np.float32), -F16_MAX, F16_MAX), nan=0.0)
        rng = np.random.default_rng(123)
        e = {f"{w}_{k}": [] for k in ("sky", "faint", "bright", "core", "chroma") for w in ("raw", "ai")}
        for i, crop in enumerate(crops):
            a, _b, t = synth_pair3(crop, rng, 4.0, 50.0, 0.7 if i % 2 else 0.0, 0.6)
            p = tiled(model, a.astype(np.float32), device)
            tl = t.mean(0)
            sc = np.sqrt(1 + np.maximum(tl, 0) / 50.0)
            for name, img in (("raw", a), ("ai", p)):
                d = ((img.mean(0) - tl) / sc) ** 2
                for key, sel in (("sky", tl < 3), ("faint", (tl >= 3) & (tl <= 15)), ("bright", (tl > 15) & (tl < 300)),
                                 ("core", tl >= 300)):
                    if sel.any():
                        e[f"{name}_{key}"].append(d[sel])
                e[f"{name}_chroma"].append((_chroma_err(img, t, tl < 3) ** 2).ravel())
        res[m["name"]] = {n: round(float(np.sqrt(np.mean(np.concatenate(v)))) if v else 0.0, 3) for n, v in e.items()}
    model.train()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/ai/n2n")
    ap.add_argument("--synth", default="output/ai/synth_rgb")
    ap.add_argument("--out", default="output/ai/model_rgb")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--width", type=int, nargs=4, default=[48, 96, 192, 256])
    ap.add_argument("--hp", type=float, default=HP_SIGMA)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = UNet3(tuple(a.width), hp_sigma=a.hp).to(device)
    (out / "config.json").write_text(json.dumps({"width": a.width, "hp": a.hp, "channels": C}), encoding="utf-8")
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    pairs = Pairs3(root, train=True)
    synth_root = Path(a.synth)
    if (synth_root / "meta.json").is_file():
        pairs.synth = Synth3(synth_root)
        print("synthetic crops", int(pairs.synth.sizes.sum()), "real groups", len({str(f) for f, _k, _s in pairs.items}),
              flush=True)
    loader = torch.utils.data.DataLoader(pairs, batch_size=a.batch, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    t0 = time.time()
    log, run, best, skipped = [], 0.0, None, 0
    for step, (x, y, kind) in enumerate(loader, start=1):
        x, y, kind = x.to(device, non_blocking=True), y.to(device, non_blocking=True), kind.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred, low = model(x, with_low=True)
        low_rms = float(low.detach().pow(2).mean().sqrt())
        loss = loss_fn(pred.float(), y, x, kind) + LOW_WEIGHT * low.pow(2).mean()
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
            print(f"step {step} loss {run:.4f} {(time.time() - t0) / step * 1000:.0f} ms/step skipped {skipped}"
                  f" low {low_rms:.2f}", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, root, device)
            if pairs.synth is not None:
                v.update(validate_synth(model, synth_root, device))
            log.append({"step": step, "loss": run, "val": v})
            print("  val:", v, flush=True)
            score = sum(d[f"ai_{k}"] / d[f"raw_{k}"] for d in v.values()
                        for k in ("sky", "faint", "bright", "core", "chroma") if d.get(f"raw_{k}", 0) > 0)
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
    dummy = torch.zeros(1, C, 256, 256, device=device)
    torch.onnx.export(model, dummy, out / "n2n_rgb.onnx", input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}, dynamo=False)
    print("exported", out / "n2n_rgb.onnx", f"total {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
