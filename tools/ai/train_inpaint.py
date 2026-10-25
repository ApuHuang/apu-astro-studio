"""訓練去星的「仿製填滿」：星點的洞用周圍的樣子補（自監督，不需要任何人的無星圖、不用付費軟體的輸出）。

資料（tools/ai/make_inpaint_data.py）：master 的亮度小塊（天空 σ 單位）＋星點遮罩＋不能挖洞的範圍。
每個樣本：隨機 256×256 → 在沒有星的地方挖 1～6 個洞（六成照真實星點遮罩的半徑、四成是 8～48 px 的大洞
（星暈）、兩成做成十字（繞射芒））→ 原本的星點也遮起來（跟去星時一樣，但不算損失）→ 洞先跟去星一樣補一次（stars.fill_holes：洞外模糊 1 px
當邊界，調和內插）當底 → 模型看「補過的影像（asinh 壓縮）＋洞的位置」，輸出對底的修正，
限制在 ±LIMIT·s（s＝局部噪聲尺度 √(1＋亮度/16)，跟降噪一樣）。損失：挖的洞裡「差模糊 1.5 px」的 L1、除以 s
（不模糊的話損失幾乎全是答案本身的噪聲，學得很慢）。bf16（fp16 第二版 2000 步後 NaN）。
第一版在 asinh 壓縮空間輸出修正、再用 sinh 還原：亮的地方（星雲、星系核心）壓縮後的值很大，修正量差一點、
還原後差好幾倍，驗證誤差比調和內插大 2～15 倍，12800 步後 NaN
洞裡的答案有噪聲，模型學到的是平滑的期望值；噪聲紋理在去星時另外補（StarSplit 的「補上噪聲紋理」）。

驗證：評估用的 master（LBN576、NGC6888、NGC2244、IC434、M31、M16）固定的洞，答案和補的結果都先模糊 1.5 px
（答案本身有噪聲）再比，跟調和內插的誤差比，分小（<8 px）、中（8～20）、大（>20）。

PYTHONPATH=src .venv-ai\\Scripts\\python tools/ai/train_inpaint.py --data output/ai/inpaint --out output/ai/inpaint_v1
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from apu_astro.studio import backend  # noqa: E402  調和內插（沒有 CuPy 時用 numpy）

CROP = 256
SOFT = 3.0
NOISE_K = 16.0
LIMIT = 8.0          # 修正量上限（局部噪聲尺度 s 的倍數）
WIDTHS = (32, 64, 128, 192, 256, 256)


def to_npy(root: Path) -> list[dict]:
    """npz 解開成 .npy（讀的時候用 mmap，6 個資料載入子行程不用各自複製一份）。"""
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    for m in meta:
        base = root / m["name"]
        if not (root / f"{m['name']}_L.npy").is_file():
            z = np.load(root / f"{m['name']}.npz")
            for k in ("L", "mask", "keepout"):
                np.save(root / f"{m['name']}_{k}.npy", z[k])
        m["n"] = int(np.load(root / f"{m['name']}_L.npy", mmap_mode="r").shape[0])
        m["base"] = str(base)
    return meta


def make_holes(keepout: np.ndarray, radii: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在 keepout 以外挖洞：圓（星點、星暈）或十字（繞射芒）。"""
    h, w = keepout.shape
    dist = ndimage.distance_transform_edt(~keepout)
    holes = np.zeros((h, w), bool)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(int(rng.integers(1, 7))):
        if rng.random() < 0.6 and len(radii):
            r = float(radii[int(rng.integers(len(radii)))]) + 2.0           # 去星時外擴 2 px
        else:
            r = float(np.exp(rng.uniform(np.log(8.0), np.log(48.0))))
        ok = (dist > r + 2) & (yy > r) & (yy < h - r) & (xx > r) & (xx < w - r)
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            continue
        c = int(cand[int(rng.integers(len(cand)))])
        cy, cx = divmod(c, w)
        shape = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        if rng.random() < 0.2:                                              # 繞射芒：十字
            ang = rng.uniform(0, np.pi / 2)
            half_w = rng.uniform(1.0, 3.0)
            length = r * rng.uniform(2.0, 4.0)
            for a in (ang, ang + np.pi / 2):
                u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
                v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
                shape |= (np.abs(u) <= length) & (np.abs(v) <= half_w)
            shape = (shape & (~keepout | ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)))
        holes |= shape
    return holes


def sample(L, mask, keepout, radii, rng, crop=CROP):
    """一個訓練樣本 → (補過的輸入, 洞＋星點遮罩, 答案, 算損失的範圍)；挖不到洞回傳 None。"""
    H, W = L.shape
    y0, x0 = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
    sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
    t = np.nan_to_num(np.asarray(L[sl], np.float32))
    m = np.asarray(mask[sl], bool)
    k = np.asarray(keepout[sl], bool)
    r = int(rng.integers(4))
    t, m, k = np.rot90(t, r), np.rot90(m, r), np.rot90(k, r)
    if rng.random() < 0.5:
        t, m, k = t[:, ::-1], m[:, ::-1], k[:, ::-1]
    t, m, k = np.ascontiguousarray(t), np.ascontiguousarray(m), np.ascontiguousarray(k)
    holes = make_holes(k, radii, rng)
    loss_m = holes & ~m
    if loss_m.sum() < 8:
        return None
    full = holes | m
    return fill_like_starsplit(t, full), full, t, loss_m


def fill_like_starsplit(t: np.ndarray, full: np.ndarray) -> np.ndarray:
    """跟去星一樣先補一次（stars.fill_holes）：洞外模糊 1 px（只用洞外的像素）當邊界，調和內插；洞外照原樣。"""
    v = (~full).astype(np.float32)
    B = ndimage.gaussian_filter(t * v, 1.0) / np.maximum(ndimage.gaussian_filter(v, 1.0), 1e-6)
    return np.where(full, backend.harmonic_fill(B[None].astype(np.float32), full)[0], t).astype(np.float32)


class Data(torch.utils.data.IterableDataset):
    def __init__(self, root: Path, seed: int = 0):
        self.meta = [m for m in to_npy(root) if not m["val"]]
        self.radii = np.load(root / "radii.npy")
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 7919 + int(time.time()))
        arrs = [{k: np.load(f"{m['base']}_{k}.npy", mmap_mode="r") for k in ("L", "mask", "keepout")} for m in self.meta]
        sizes = np.array([m["n"] for m in self.meta], np.float64)
        while True:
            a = arrs[rng.choice(len(arrs), p=sizes / sizes.sum())]
            i = int(rng.integers(len(a["L"])))
            s = sample(a["L"][i], a["mask"][i], a["keepout"][i], self.radii, rng)
            if s is None:
                continue
            u = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))   # 噪聲尺度量得不準一點
            filled, full, t, loss_m = s
            yield (torch.from_numpy(filled * u)[None], torch.from_numpy(full.astype(np.float32))[None],
                   torch.from_numpy(t * u)[None], torch.from_numpy(loss_m.astype(np.float32))[None])


# ---------------------------------------------------------------------- 網路


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))


def asinh(v: torch.Tensor) -> torch.Tensor:
    return torch.sign(v) * torch.log(torch.abs(v) + torch.sqrt(v * v + 1))


class InpaintNet(nn.Module):
    """輸入 (補過的影像 σ 單位, 遮罩)，輸出補好的影像（遮罩外不變）。邊長要是 32 的倍數。"""

    def __init__(self, ch: tuple[int, ...] = WIDTHS):
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

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            comp = asinh(x.float() / SOFT)
        z = torch.cat([comp, m.float()], 1)
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
            s = torch.sqrt(1 + torch.clamp(x.float(), min=0) / NOISE_K)
            pred = x.float() + s * LIMIT * torch.tanh(r / LIMIT)
            return torch.where(m > 0.5, pred, x.float())


_KERNEL: dict = {}


def blur(v: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """高斯模糊（3σ 的核，邊上反射）。"""
    key = (v.device, sigma)
    if key not in _KERNEL:
        r = int(np.ceil(3 * sigma))
        g = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
        _KERNEL[key] = (g / g.sum()).to(v.device)
    k = _KERNEL[key]
    r = (len(k) - 1) // 2
    v = F.pad(v, (r, r, r, r), mode="reflect")
    v = F.conv2d(v, k.view(1, 1, 1, -1))
    return F.conv2d(v, k.view(1, 1, -1, 1))


def loss_fn(pred, target, x, loss_m):
    """挖的洞裡的 L1，差先模糊 1.5 px 再算：答案（原圖）每個像素都有噪聲，不模糊的話損失幾乎全是噪聲
    （逐像素 0.9 裡能學的只有 0.01，學得很慢）。洞外 pred＝x＝原圖，差是 0，模糊時不會把洞外帶進來。"""
    s = torch.sqrt(1 + torch.clamp(x, min=0) / NOISE_K)
    d = blur((pred - target).float())
    return (torch.abs(d) / s * loss_m).sum() / loss_m.sum().clamp(min=1)


# ---------------------------------------------------------------------- 驗證


def make_val(root: Path, n_per: int = 12) -> list:
    rng = np.random.default_rng(99)
    radii = np.load(root / "radii.npy")
    out = []
    for m in to_npy(root):
        if not m["val"]:
            continue
        L = np.load(f"{m['base']}_L.npy", mmap_mode="r")
        M = np.load(f"{m['base']}_mask.npy", mmap_mode="r")
        K = np.load(f"{m['base']}_keepout.npy", mmap_mode="r")
        got = 0
        for i in range(len(L)):
            s = sample(L[i], M[i], K[i], radii, rng, crop=256)
            if s is not None:
                out.append((m["name"], *s))
                got += 1
            if got >= n_per:
                break
    return out


def validate(model, val, device) -> dict:
    model.eval()
    bins = {"small": (0, 8), "mid": (8, 20), "big": (20, 999)}
    err = {f"{w}_{b}": [] for w in ("ai", "harm") for b in bins}
    with torch.no_grad():
        for _name, filled, full, t, loss_m in val:
            x = torch.from_numpy(filled)[None, None].to(device)
            m = torch.from_numpy(full.astype(np.float32))[None, None].to(device)
            p = model(x, m)[0, 0].float().cpu().numpy()
            s = np.sqrt(1 + np.maximum(filled, 0) / NOISE_K)
            tb = ndimage.gaussian_filter(t, 1.5)
            lab, n = ndimage.label(loss_m)
            sizes = ndimage.sum(np.ones_like(t), lab, range(1, n + 1))
            for name, img in (("ai", p), ("harm", filled)):
                d = np.abs(ndimage.gaussian_filter(img, 1.5) - tb) / s
                for j in range(n):
                    reg = lab == j + 1
                    r_eq = math.sqrt(sizes[j] / math.pi)
                    for b, (lo, hi) in bins.items():
                        if lo <= r_eq < hi:
                            err[f"{name}_{b}"].append(float(d[reg].mean()))
    model.train()
    return {k: round(float(np.mean(v)), 3) if v else None for k, v in err.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/ai/inpaint")
    ap.add_argument("--out", default="output/ai/inpaint_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--init", help="從這個權重接著練（優化器、學習率從頭排；例如上一次中斷的 inpaint.pt）")
    ap.add_argument("--warmup", type=float, default=0.05, help="學習率爬升占總步數的比例")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = InpaintNet().to(device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device))
        print("init from", a.init, flush=True)
    data = Data(root)
    print("train masters", len(data.meta), "crops", sum(m["n"] for m in data.meta), "radii", len(data.radii), flush=True)
    val = make_val(root)
    print("val holes from", len(val), "crops", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.warmup)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    t0, run, run_h, best, log = time.time(), 0.0, 0.0, None, []
    # 卡住的保護：第二版在學習率最高時（1400 步）輸出整片頂到上限，tanh 飽和、梯度幾乎是 0，卡到 9400 步學習率降下來
    # 才自己出來（一半的步數白費）。損失比調和內插大 1.5 倍就退回最近一次正常的權重、清掉 Adam 的動量、學習率減半
    good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lr_scale = 1.0
    for step, (x, m, t, lm) in enumerate(loader, start=1):
        x, m, t, lm = (v.to(device, non_blocking=True) for v in (x, m, t, lm))
        # bf16：fp16 幾千步後溢位變 NaN（第二版 2000 步）；bf16 範圍跟 fp32 一樣，不用 GradScaler
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, m)
        loss = loss_fn(pred, t, x, lm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for g in opt.param_groups:
            g["lr"] *= lr_scale
        with torch.no_grad():
            lh = float(loss_fn(x, t, x, lm))                                   # 調和內插的損失（對照）
        lv = float(loss.detach())
        run = 0.98 * run + 0.02 * lv if step > 1 else lv
        run_h = 0.98 * run_h + 0.02 * lh if step > 1 else lh
        if step % 200 == 0:
            print(f"step {step} loss {run:.4f} (harmonic {run_h:.4f}) {(time.time() - t0) / step * 1000:.0f} ms/step",
                  flush=True)
            if run > 1.5 * run_h:
                model.load_state_dict(good)
                opt.state.clear()
                lr_scale *= 0.5
                run = run_h
                print(f"  stuck: back to last good weights, lr x{lr_scale}", flush=True)
            elif run <= 1.05 * run_h and lv <= 1.5 * lh:
                good = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, val, device)
            log.append({"step": step, "val": v})
            print("  val:", v, flush=True)
            score = sum(v[f"ai_{b}"] / v[f"harm_{b}"] for b in ("small", "mid", "big") if v[f"harm_{b}"])
            if best is None or score < best[0]:
                best = (score, step)
                torch.save(model.state_dict(), out / "inpaint.pt")
                print(f"  best so far (score {score:.3f})", flush=True)
            (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if step >= a.steps:
            break
    model.load_state_dict(torch.load(out / "inpaint.pt"))
    print("export best step", best[1], flush=True)
    model.eval().float()
    dx = torch.zeros(1, 1, 256, 256, device=device)
    torch.onnx.export(model, (dx, dx), out / "inpaint.onnx", input_names=["x", "mask"], output_names=["y"],
                      opset_version=17, dynamic_axes={"x": {2: "h", 3: "w"}, "mask": {2: "h", 3: "w"},
                                                      "y": {2: "h", 3: "w"}}, dynamo=False)
    print("exported", out / "inpaint.onnx", f"total {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
