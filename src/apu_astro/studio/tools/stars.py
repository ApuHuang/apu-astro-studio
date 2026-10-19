"""去星（分成無星圖＋星點層）與合成星點。

去星（傳統方法，之後可以換成 AI 模型）：找星點 → 星點範圍用周圍的值補（由小到大的正規化卷積）→ 補的地方加上
跟附近一樣的噪聲紋理（不然無星圖上會有一個個太光滑的圓斑）→ 星點層＝原圖 − 無星圖。
在線性階段做，之後兩層分開拉伸，最後合成星點時加回去。
"""

from __future__ import annotations

import numpy as np
from apu_processing import detail as pdetail
from scipy import ndimage

from .. import backend, ops
from . import T, Ctx, State, Tool, choice, register, slider, toggle


SMOOTH = 7      # 沿方向平均的長度（px）：繞射芒、星暈外圍每個像素都比噪聲暗，要平均才量得到


def _ray(hp: np.ndarray, cy: float, cx: float, angle: float, t: np.ndarray, half: float) -> np.ndarray:
    """沿一個方向取樣（垂直方向取 3 點、沿方向 SMOOTH 點平均），回傳每個距離的值。"""
    h, w = hp.shape
    dy, dx = np.sin(angle), np.cos(angle)
    acc = np.zeros(len(t), np.float64)
    for off in (-half, 0.0, half):
        ys = np.clip(cy + t * dy + off * dx, 0, h - 1)
        xs = np.clip(cx + t * dx - off * dy, 0, w - 1)
        acc += ndimage.map_coordinates(hp, [ys, xs], order=1)
    return ndimage.uniform_filter1d(acc / 3, SMOOTH, mode="nearest")


def _background(L: np.ndarray, f: int = 8) -> np.ndarray:
    """不受星點影響的背景：f×f 區塊取中位數 → 再取 7×7 中位數 → 放大回原尺寸。
    （用高斯模糊當背景的話，亮星附近的背景會被星點拉高，星暈、繞射芒反而變成負的）"""
    h, w = L.shape
    h2, w2 = max(f, h // f * f), max(f, w // f * f)
    if h < f or w < f:
        return np.full_like(L, float(np.median(L)))
    blocks = L[:h2, :w2].reshape(h2 // f, f, w2 // f, f).transpose(0, 2, 1, 3).reshape(h2 // f, w2 // f, -1)
    small = ndimage.median_filter(np.median(blocks, axis=2), size=7, mode="nearest")
    zy = np.clip((np.arange(h) + 0.5) / f - 0.5, 0, small.shape[0] - 1)
    zx = np.clip((np.arange(w) + 0.5) / f - 0.5, 0, small.shape[1] - 1)
    yy, xx = np.meshgrid(zy, zx, indexing="ij")
    return ndimage.map_coordinates(small, [yy, xx], order=1).astype(np.float32)


def _extent(profile: np.ndarray, t: np.ndarray, limit: float) -> float:
    """亮度降到 limit 以下（連續 6 個點）之前的最遠距離。"""
    run = 0
    last = 0.0
    for i, a in enumerate(profile > limit):
        if a:
            run = 0
            last = float(t[i])
        else:
            run += 1
            if run >= 6:
                break
    return last


NOT_GIVEN = object()


def estimate_spike_angle(L: np.ndarray, stars, scale: float, hp: np.ndarray | None = None,
                         bright: np.ndarray | None = None) -> int | None:
    """繞射芒方向（度，0～89）：最大的 15 顆亮星在環狀區域量各方向的亮度，找出四個相隔 90° 的方向；
    明顯比其他方向亮、而且四支都在才算（折射鏡沒有；星雲纖維只有兩支）。沒有就 None。
    最好在原圖上量（反摺積後星點變小、AI 降噪後細的芒變淡，對比會掉到門檻邊緣）。"""
    h, w = L.shape
    if hp is None:
        hp = (L - _background(L)).astype(np.float32)
    if bright is None:
        sample = hp[:: max(1, h // 512), :: max(1, w // 512)]
        sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
        peak = np.array([hp[int(round(y)), int(round(x))] for y, x in zip(np.clip(stars.y, 0, h - 1),
                                                                             np.clip(stars.x, 0, w - 1))])
        bright = np.flatnonzero(peak > 30 * sigma)
    if len(bright) == 0:
        return None
    bright = bright[np.argsort(-stars.radius[bright])]           # 用大小排（飽和星的峰值都一樣、反摺積會改變峰值）
    angle = None
    angles = np.radians(np.arange(0, 360, 1.0))
    prof = np.zeros(360)
    used = 0
    for i in bright[:15]:
        r0 = max(float(stars.radius[i]), 2.0)
        cy, cx = float(stars.y[i]), float(stars.x[i])
        # 取樣環有固定的下限（完整尺寸 8～40 px）：反摺積後星點變小，只用星點半徑的倍數會太靠近星核，被星暈蓋過
        radii = np.linspace(max(2.0 * r0, 8.0 / max(scale, 1e-6)), max(8.0 * r0, 40.0 / max(scale, 1e-6)), 24)
        ys = cy + np.outer(radii, np.sin(angles))
        xs = cx + np.outer(radii, np.cos(angles))
        if ((ys >= 0) & (ys < h - 1) & (xs >= 0) & (xs < w - 1)).mean() < 0.9:
            continue
        vals = ndimage.map_coordinates(hp, [np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], order=1)
        p = np.median(vals, axis=0)
        p = p - np.median(p)
        prof += p / (np.abs(p).max() or 1.0)
        used += 1
    if used >= 3:
        four = prof[:90] + prof[90:180] + prof[180:270] + prof[270:]
        best = int(np.argmax(four))
        mad = 1.4826 * np.median(np.abs(four - np.median(four))) + 1e-9
        # 四支都要在：星雲纖維、亮邊只有沿著它的兩支（垂直方向是暗的），折疊後也會冒出一個方向
        # （NGC6888 是折射鏡，亮星都在斜的星雲亮邊上，曾經沿亮邊畫出繞射芒遮罩，整條亮邊被補掉）
        base_level = float(np.median(prof))
        arms = [float(np.max(prof[(best + 90 * k - 3) % 360:(best + 90 * k - 3) % 360 + 7]))
                if (best + 90 * k - 3) % 360 + 7 <= 360 else float(np.max(np.roll(prof, -((best + 90 * k - 3) % 360))[:7]))
                for k in range(4)]
        excess = np.array(arms) - base_level
        if (four[best] - np.median(four)) / mad >= 6 and excess.min() > 0.4 * excess.max():
            angle = best

    return angle


HALO_INNER = 1.5    # 星暈剩多少（像素噪聲的幾倍）以內用扣的，更裡面挖掉補（見 _halo_profile）


def _halo_profile(planes: np.ndarray, cy: float, cx: float, base: float, r_core: float, noise: float,
                  scale: float, pixel: float):
    """一顆亮星的星暈剖面（每個色版）：8 個避開繞射芒的方向取中位數，從星點範圍往外只准變暗（累積最小值），
    到了不再變暗的地方（往外 20 px 降不到半個噪聲；星暈的盡頭或星雲的平台）為止，扣掉那裡的值當底。
    回傳 (距離, 每色版的星暈 (C, n), 內圈半徑, 外圈半徑) 或 None。noise：剖面（8 個方向的中位數）的噪聲；
    pixel：像素噪聲。內圈＝星暈降到 HALO_INNER 倍像素噪聲的地方：更裡面星點不完全對稱（反摺積、繞射芒、色差，
    同一個半徑有的方向 20σ、有的 0），用對稱的剖面扣會在暗的方向挖出洞，挖掉補。"""
    px = 1.0 / max(scale, 1e-6)
    t = np.arange(0, max(30 * px, 10 * r_core), 1.0)
    h, w = planes.shape[1:]
    raw, allrays = [], []
    for c in planes:
        rays = []
        for k in range(8):
            a = base + k * np.pi / 4
            dy, dx = np.sin(a), np.cos(a)
            acc = np.zeros(len(t))
            for off in (-1.0, 0.0, 1.0):
                ys = np.clip(cy + t * dy + off * dx, 0, h - 1)
                xs = np.clip(cx + t * dx - off * dy, 0, w - 1)
                acc += ndimage.map_coordinates(c, [ys, xs], order=1)
            rays.append(acc / 3)
        raw.append(np.median(rays, axis=0))
        allrays.append(rays)
    # 沿方向平均的長度隨距離變長（距離的 1/4）：靠近星核不能平均（_ray 的固定 7 px 會把星核的光平均進來，
    # 小星的剖面量得太高、多扣，每顆星旁邊一個暗洞），外圍很淡要平均才量得到
    cs = np.concatenate([np.zeros((len(raw), 1)), np.cumsum(raw, axis=1)], axis=1)
    half = np.maximum(0, np.round(0.125 * t)).astype(int)
    lo = np.clip(np.arange(len(t)) - half, 0, len(t) - 1)
    hi = np.clip(np.arange(len(t)) + half + 1, 1, len(t))
    prof = (cs[:, hi] - cs[:, lo]) / (hi - lo)
    # 8 個方向彼此差多少（同樣平滑過）：星暈各方向差不多，星雲、別的星各方向差很多
    rl = np.mean(allrays, axis=0)                                            # (8, T)
    cr = np.concatenate([np.zeros((8, 1)), np.cumsum(rl, axis=1)], axis=1)
    rl = (cr[:, hi] - cr[:, lo]) / (hi - lo)
    spread = 1.4826 * np.median(np.abs(rl - np.median(rl, axis=0)), axis=0)
    j0 = min(int(np.ceil(r_core)), len(t) - 1)
    m = np.minimum.accumulate(prof[:, j0:], axis=1)
    mL = m.mean(axis=0)
    gap = max(1, int(round(20 * px)))
    if len(mL) <= gap + 1:
        return None
    # 盡頭：往外 20 px 降不到半個噪聲，或降不到現在高度（比整條最低的地方高多少）的 15%。真的星暈越往外
    # 越暗得快（冪次），星雲是平台：只看前者的話星雲裡剖面一直在噪聲以上往下走，NGC6888 的星雲核心
    # 被當成星暈扣掉、內圈挖出好幾塊大圓斑
    drop = mL[:-gap] - mL[gap:]
    height = mL[:-gap] - mL[-1]
    # 而且星暈要比各方向的差異明顯（2 倍）：不然量到的是星雲，不是這顆星的光（NGC6888 星雲核心）
    messy = height < 2 * spread[j0:j0 + len(height)]
    flat = np.flatnonzero((drop < 0.5 * noise) | (drop < 0.15 * height) | messy)
    j_end = int(flat[0]) if len(flat) else len(mL) - gap - 1
    if j_end <= 0:
        return None
    # 底：盡頭外面那一段的中位數（累積最小值會被噪聲一路往下拉，拿它當底的話整個星暈多扣，星的位置變暗洞）
    floor = np.median(prof[:, j0 + j_end:j0 + j_end + gap + 1], axis=1, keepdims=True)
    halo = np.clip(m[:, :j_end + 1] - floor, 0, None)
    if halo.mean(axis=0)[0] < 5 * noise:                   # 星點範圍外面就沒有星暈了
        return None
    inner = np.flatnonzero(halo.mean(axis=0) < HALO_INNER * pixel)
    j_in = min(int(inner[0]) if len(inner) else j_end, int(4 * r_core))     # 內圈最多星點範圍的 5 倍
    return t[j0:j0 + j_end + 1], halo, float(t[j0 + j_in]), float(t[j0 + j_end])


def halo_spike_mask(L: np.ndarray, stars, scale: float, noise_boost: float = 1.0,
                    core_r: np.ndarray | None = None, angle=NOT_GIVEN, planes: np.ndarray | None = None):
    """亮星的星暈與繞射芒（星點偵測只抓到星心，亮星外圍的光暈、十字芒會留在無星圖上）。

    1. 繞射芒方向：angle（原圖上量好的，見 estimate_spike_angle）；沒給就在這裡量
    2. 每顆亮星：八個避開繞射芒的方向取中位數，量亮度降到噪聲以下的距離 → 星暈半徑；
       沿繞射芒方向量（扣掉同距離的星暈亮度）→ 每支芒的長度。暗星量出來是 0，不會多遮
    門檻用隨機位置量同樣的取樣得到的噪聲（影像的噪聲是相關的，不能用理論值）。
    noise_boost：前面的 AI 降噪把噪聲降了幾倍（State.notes 的 noise_ratio）；門檻乘回去，範圍才跟不用 AI 時一樣
    （不乘的話噪聲變小、星暈量得更遠，補的範圍變很大，無星圖上留下一大片不規則的補痕）
    3. 有給 planes 和 core_r 時，星暈改用扣的（_halo_profile）：只挖掉內圈，外面扣掉量到的星暈剖面。
       以前整個星暈挖掉補：星暈其實比量到的範圍遠，邊界上還有星暈的光，補出來是一塊比四周亮的圓盤
       （LBN576 的亮星），而且密集星場一大片被補掉、星雲紋理沒了。
    回傳 (遮罩, 繞射芒方向, 要扣掉的星暈 (C, H, W) 或 None)。
    """
    if len(stars.y) == 0:
        return None, None, None
    h, w = L.shape
    hp = (L - _background(L)).astype(np.float32)
    sample = hp[:: max(1, h // 512), :: max(1, w // 512)]
    sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) * noise_boost
    peak = np.array([hp[int(round(y)), int(round(x))] for y, x in zip(np.clip(stars.y, 0, h - 1),
                                                                         np.clip(stars.x, 0, w - 1))])
    # 亮星：星心比背景亮 30σ 以上（最多 400 顆）
    bright = np.flatnonzero(peak > 30 * sigma)
    bright = bright[np.argsort(-peak[bright])][:400]
    if len(bright) == 0:
        return None, None, None

    if angle is NOT_GIVEN:
        angle = estimate_spike_angle(L, stars, scale, hp, bright)

    # 噪聲：隨機位置做一樣的取樣
    rng = np.random.default_rng(7)
    t_noise = np.arange(0, 40, 1.0)
    singles, medians = [], []
    for _ in range(120):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        a0 = rng.uniform(0, 2 * np.pi)
        rays = [_ray(hp, cy, cx, a0 + k * np.pi / 4, t_noise, 1.0) for k in range(8)]
        singles.append(rays[0][SMOOTH:-SMOOTH])
        medians.append(np.median(rays, axis=0)[SMOOTH:-SMOOTH])

    def mad_of(v: list) -> float:
        v = np.concatenate(v)
        return 1.4826 * float(np.median(np.abs(v - np.median(v))))

    ray_noise = mad_of(medians)
    halo_limit = 3.0 * ray_noise * noise_boost
    spike_limit = 3.0 * mad_of(singles) * noise_boost

    mask = np.zeros((h, w), bool)
    model = planes is not None and core_r is not None
    halos = np.zeros(planes.shape, np.float32) if model else None
    pixel = ops.pixel_noise(L)
    # 從最亮的開始，量完一顆就從工作影像扣掉再量下一顆：不然大星星暈裡的每顆星都把大星的星暈當成自己的，
    # 重複扣好幾次，亮星周圍變成黑洞（LBN576）
    work = planes.astype(np.float32, copy=True) if model else None
    yy, xx = np.ogrid[:h, :w]
    max_len = 600.0 / max(scale, 1e-6)
    base = np.radians((angle if angle is not None else 0) + 22.5)
    for i in bright:
        r0 = max(float(stars.radius[i]), 1.5)
        cy, cx = float(stars.y[i]), float(stars.x[i])
        t = np.arange(r0, max_len, 1.0)
        ref = np.median([_ray(hp, cy, cx, base + k * np.pi / 4, t, 1.0) for k in range(8)], axis=0)
        # 星暈到哪裡：亮度降到噪聲以下，或剖面不再下降（到了星雲的平台；星雲裡亮度一直在噪聲以上，
        # 只看前者的話半徑一路量到上限，NGC6888 的星雲亮邊整片被補掉）。上限＝這顆星範圍的 6 倍
        ext = _extent(ref, t, halo_limit)
        decline = ref[:-4] - ref[4:] if len(ref) > 4 else np.zeros(0)
        flat = np.flatnonzero(decline < halo_limit * 0.5)
        if len(flat):
            ext = min(ext, float(t[flat[0]]))
        if model:
            got = _halo_profile(work, cy, cx, base, float(core_r[i]), ray_noise, scale, pixel)
            draw = False
            halo = float(core_r[i])
            if got is not None:
                tt, hc, r_in, r_out = got
                halo = max(r_in, halo)
                y0, y1 = int(max(0, cy - r_out - 1)), int(min(h, cy + r_out + 2))
                x0, x1 = int(max(0, cx - r_out - 1)), int(min(w, cx + r_out + 2))
                rr = np.sqrt((yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2)
                for c in range(len(hc)):
                    one = np.interp(rr, tt, hc[c], left=hc[c][0], right=0.0).astype(np.float32)
                    halos[c, y0:y1, x0:x1] += one
                    work[c, y0:y1, x0:x1] -= one
                if r_in > float(core_r[i]):
                    draw = True
        elif core_r is not None:
            # 只有星暈超出這顆星自己的範圍才畫（以前下限是找星的連通區半徑＋2 px，每顆中等亮度的星都多一圈，
            # NGC6888 的密集星場星暈遮罩就佔了三分之一）
            halo = min(ext * 1.15 + 1.0 / max(scale, 1e-6), 6.0 * float(core_r[i]))
            if halo <= float(core_r[i]):
                halo = float(core_r[i])
                draw = False
            else:
                draw = True
        else:
            halo = min(max(ext * 1.15, r0) + 2.0, max(120.0 / max(scale, 1e-6), r0 + 2.0))
            draw = True
        if draw:
            y0, y1 = int(max(0, cy - halo - 1)), int(min(h, cy + halo + 2))
            x0, x1 = int(max(0, cx - halo - 1)), int(min(w, cx + halo + 2))
            mask[y0:y1, x0:x1] |= (yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2 <= halo * halo
        if angle is None:
            continue
        width = max(1.5, 0.4 * r0)
        for k in range(4):
            a = np.radians(angle + 90 * k)
            length = _extent(_ray(hp, cy, cx, a, t, width) - np.maximum(ref, 0), t, spike_limit)
            if length <= halo:
                continue
            tt = np.arange(0, length * 1.3 + SMOOTH, 0.5)
            for off in np.arange(-width - 1.5, width + 1.51, 0.5):
                ys = np.round(cy + tt * np.sin(a) + off * np.cos(a)).astype(int)
                xs = np.round(cx + tt * np.cos(a) - off * np.sin(a)).astype(int)
                ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                mask[ys[ok], xs[ok]] = True
    return (mask if mask.any() else None), angle, halos


def star_footprints(L: np.ndarray, stars, scale: float, grow: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """每顆星自己的範圍：從星心往 8 個方向取亮度剖面（取第二暗的方向），扣掉外圈的局部背景，
    降到峰值 10% 的半徑 × 1.6＋1 px（但不超過半高半徑 × 2.5＋1.5、也不超過降到噪聲以下的半徑＋1.5）
    再加 grow 畫成圓（px 都是完整尺寸）。暗星的 10% 在噪聲裡量不準、常常量得很大：以前暗星一顆半徑 7 px 以上，
    門檻放低一點遮罩就蓋掉半張圖（NGC6888），只好把門檻設高，結果暗星留在無星圖上、拉伸後很明顯。
    回傳 (遮罩, 每顆星的半徑，不含 grow)。
    不用找星的連通區域：亮星常跟旁邊的星雲連成一塊，整塊被當成星（NGC6888 星雲亮邊被搬進星點層）。
    半徑、圓心都用小數：快速預覽的一個像素是完整尺寸的 4 px，取整數的話每顆星大一圈，預覽補掉的比成品多。"""
    h, w = L.shape
    n = len(stars.y)
    mask = np.zeros((h, w), bool)
    if n == 0:
        return mask, np.zeros(0)
    rmax = max(6, int(round(24 / max(scale, 1e-6))))
    step = 1.0 / max(1, int(round(scale)))                 # 取樣間隔＝完整尺寸 1 px
    rr = np.arange(0, rmax + 1e-6, step, dtype=np.float32)
    th = np.radians(np.arange(0, 360, 45, dtype=np.float32))
    radii = np.empty(n, np.float32)
    half = np.empty(n, np.float32)
    outer = rr >= 0.7 * rmax
    noise = ops.pixel_noise(L)
    lost = np.empty(n, np.float32)
    for a in range(0, n, 4000):                      # 一次 4000 顆，省記憶體
        y = stars.y[a:a + 4000, None, None].astype(np.float32)
        x = stars.x[a:a + 4000, None, None].astype(np.float32)
        ys = np.clip(y + rr[None, None, :] * np.sin(th)[None, :, None], 0, h - 1)
        xs = np.clip(x + rr[None, None, :] * np.cos(th)[None, :, None], 0, w - 1)
        prof = ndimage.map_coordinates(L, [ys.ravel(), xs.ravel()], order=1).reshape(ys.shape)
        # 第二暗的方向：星在星雲纖維上時，只有垂直纖維的方向會降到真正的背景（用中位數的話纖維把半徑撐大）
        prof = np.sort(prof, axis=1)[:, 1, :]                          # (k, r)
        bg = np.median(prof[:, outer], axis=1, keepdims=True)
        pk = np.maximum(prof[:, :1] - bg, 1e-12)
        rel = (prof - bg) / pk
        below = rel < 0.10
        radii[a:a + 4000] = np.where(below.any(axis=1), below.argmax(axis=1) * step, rmax)
        b50 = rel < 0.5
        half[a:a + 4000] = np.where(b50.any(axis=1), b50.argmax(axis=1) * step, rmax)
        gone = (prof - bg) < noise
        lost[a:a + 4000] = np.where(gone.any(axis=1), gone.argmax(axis=1) * step, rmax)
    # 半高半徑也限制一下：嵌在星雲裡的星外圈背景不平，降不到 10%，只看 r10 會一路量到上限
    px = 1.0 / max(scale, 1e-6)                            # 完整尺寸 1 px 在這個解析度是幾個像素
    R = np.clip(np.minimum(np.minimum(1.6 * radii + px, 2.5 * half + 1.5 * px), lost + 1.5 * px), 2.0 * px, 20.0 * px)
    for y, x, r in zip(stars.y, stars.x, R + grow * px):
        ri = int(np.ceil(r)) + 1
        yi, xi = int(round(y)), int(round(x))
        y0, y1, x0, x1 = max(0, yi - ri), min(h, yi + ri + 1), max(0, xi - ri), min(w, xi + ri + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return mask, R


SHARPNESS = 0.2     # 星點的尖銳度門檻（見 sharpness）


def sharp_limit(scale: float) -> float:
    """這個解析度的尖銳度門檻。快速預覽（縮小 2 倍以上）的星點只剩 1～2 px，分不出星點和亮結
    （NGC6888：完整解析度判定為星的，縮圖上 5% 只有 0.08），門檻放低到幾乎不擋；曾經縮圖上所有的星都不及格，
    快速預覽完全沒去星。完整解析度、預覽區照常。"""
    return SHARPNESS if scale <= 1.5 else 0.05


def sharpness(L: np.ndarray, scale: float, hp: np.ndarray | None = None) -> np.ndarray:
    """尖銳度：跟周圍（模糊 1.5 px）差多少 ÷ 比背景亮多少。高斯星點（σ 1.5 px）約 0.5，σ 3 px 約 0.2，
    星雲的亮結（σ 5 px 以上）0.08 以下。用來分辨星點和星雲的亮結（NGC6888 的星雲亮邊被當成星點搬進星點層）。"""
    if hp is None:
        hp = L - _background(L)
    fine = L - ops.blur(L, max(1.5 / max(scale, 1e-6), 1.0))   # 縮圖上 1.5 px 不到一個像素，模糊等於沒做
    return fine / np.maximum(hp, 1e-12)


ANISOTROPY = 0.35   # 星點對稱性門檻（見 star_like）


def _radial_profiles(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, rmax: int) -> np.ndarray:
    """每顆星往 8 個方向（0°、45°…315°）的亮度剖面 (n, 8, rmax+1)。"""
    h, w = L.shape
    rr = np.arange(0, rmax + 1, dtype=np.float32)
    th = np.radians(np.arange(0, 360, 45, dtype=np.float32))
    out = np.empty((len(ys), 8, rmax + 1), np.float32)
    for a in range(0, len(ys), 4000):
        y = ys[a:a + 4000, None, None].astype(np.float32)
        x = xs[a:a + 4000, None, None].astype(np.float32)
        py = np.clip(y + rr[None, None, :] * np.sin(th)[None, :, None], 0, h - 1)
        px = np.clip(x + rr[None, None, :] * np.cos(th)[None, :, None], 0, w - 1)
        out[a:a + 4000] = ndimage.map_coordinates(L, [py.ravel(), px.ravel()], order=1).reshape(py.shape)
    return out


def anisotropy(L: np.ndarray, stars, scale: float) -> np.ndarray:
    """星點的「二重不對稱」：在半高半徑兩倍的地方，0°／90° 與 45°／135° 兩組互相垂直的方向（各取正反兩邊平均）
    差多少（相對峰值）。星點各方向一樣；十字繞射芒四重對稱，兩組垂直方向一樣亮，也是 0；
    星雲纖維上的亮結沿纖維亮、垂直方向暗 → 很大（NGC6888 亮邊上的偵測中位數 0.18、九成 1.1，天空的星 0.06／0.23）。"""
    if len(stars.y) == 0:
        return np.zeros(0)
    rmax = max(6, int(round(24 / max(scale, 1e-6))))
    prof = _radial_profiles(L, stars.y, stars.x, rmax)
    med = np.median(prof, axis=1)
    bg = np.median(med[:, int(0.7 * rmax):], axis=1)
    pk = np.maximum(med[:, 0] - bg, 1e-12)
    rel = (med - bg[:, None]) / pk[:, None]
    r50 = np.where((rel < 0.5).any(axis=1), (rel < 0.5).argmax(axis=1), rmax)
    r2 = np.clip(2 * r50, 1, rmax)
    at = prof[np.arange(len(stars.y)), :, r2] - bg[:, None]            # (n, 8)
    axis = at[:, :4] + at[:, 4:]                                         # 正反兩邊：0°, 45°, 90°, 135°
    a = (np.abs(axis[:, 0] - axis[:, 2]) + np.abs(axis[:, 1] - axis[:, 3])) / (2 * pk)
    # 噪聲造成的不對稱（暗星的剖面很吵）：扣掉 3 倍，暗星不會被噪聲誤判成星雲
    noise = ops.pixel_noise(L)
    return a - 3 * 1.4 * noise / pk


def flat_tops(hp: np.ndarray, scale: float) -> np.ndarray:
    """飽和的平頂星核：附近（±6 px）最高點的 90% 以上、3×3 都在 85% 以上，而且比背景亮 50σ。
    疊圖後的飽和星核不太平（每張 sub 截在不同高度、對齊時插值），LBN576 一顆亮星中間 3×3 差 9%；
    沒飽和的星要 σ 3 px 以上才會這麼平（那也是星，star_like 另外要求各方向對稱，星雲的亮結過不了）。
    跟自己附近比、不跟整張的最大值比：校色、反摺積、AI 之後每顆飽和星的高度不一樣（LBN576 預覽區裡
    一顆平頂在 1.19 的亮星，因為區域裡別處有 2.08 就不算飽和，又因為平頂不尖銳被當成星雲留在無星圖上）。"""
    sample = hp[:: max(1, hp.shape[0] // 512), :: max(1, hp.shape[1] // 512)]
    sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) + 1e-12
    size = 2 * int(np.ceil(6.0 / max(scale, 1e-6))) + 1
    top = ndimage.maximum_filter(hp, size=max(size, 3))
    low = ndimage.minimum_filter(hp, size=3)
    return (hp >= 0.90 * top) & (low >= 0.85 * top) & (top > 50 * sigma)


def star_like(L: np.ndarray, stars, scale: float) -> np.ndarray:
    """找星結果裡每顆星是不是真的星：各方向對稱（二重不對稱 < ANISOTROPY），而且夠尖銳（≥ SHARPNESS）
    或是飽和的平頂（平頂不尖銳）。回傳 bool 陣列。"""
    if len(stars.y) == 0:
        return np.zeros(0, bool)
    hp = L - _background(L)
    fine = L - ops.blur(L, max(1.5 / max(scale, 1e-6), 1.0))
    h, w = L.shape
    fmax = ndimage.maximum_filter(fine, size=3)
    hmax = ndimage.maximum_filter(hp, size=3)
    y = np.clip(np.round(stars.y).astype(int), 0, h - 1)
    x = np.clip(np.round(stars.x).astype(int), 0, w - 1)
    sharp = fmax[y, x] / np.maximum(hmax[y, x], 1e-12) >= sharp_limit(scale)
    round_ = anisotropy(L, stars, scale) < ANISOTROPY
    # 飽和的平頂星核尖銳度接近 0：用平頂判斷（找星自己的飽和旗標只認得整張最亮的那顆）
    flat = ndimage.binary_dilation(flat_tops(hp, scale), iterations=1)
    sat = flat[y, x] | stars.saturated
    return round_ & (sharp | sat)


def recenter_stars(L: np.ndarray, stars, scale: float):
    """星心改到每一塊裡真正的亮點（小數位置），一塊裡有好幾個亮點（至少最亮的 25%）就拆成好幾顆。
    找星給的是整塊的重心：亮星跟旁邊的星、星雲連成一塊時重心偏掉（LBN576 一顆中等亮度的星偏了 7 px），
    遮罩畫在旁邊，星本身整顆留在無星圖上。"""
    import dataclasses
    if len(stars.y) == 0:
        return stars
    Ls = ops.blur(L, max(0.7, 1.0 / max(scale, 1e-6)))
    lab, n = ndimage.label(stars.mask)
    if n != len(stars.y):                  # 對不上（不該發生）：照原樣
        return stars
    size = 2 * max(2, int(round(3.0 / max(scale, 1e-6)))) + 1
    peaks = (Ls == ndimage.maximum_filter(Ls, size=size)) & stars.mask
    py, px = np.nonzero(peaks)
    comp = lab[py, px]
    cmax = np.asarray(ndimage.maximum(Ls, lab, index=np.arange(n + 1)))
    ok = Ls[py, px] >= 0.25 * cmax[comp]
    py, px, comp = py[ok], px[ok], comp[ok]
    has = np.zeros(n + 1, bool)
    has[comp] = True
    h, w = L.shape
    yy, xx = np.empty(len(py)), np.empty(len(py))
    for k, (y, x) in enumerate(zip(py, px)):           # 3×3 加權重心（小數位置）
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        win = Ls[y0:y1, x0:x1]
        wt = win - win.min()
        tot = float(wt.sum()) or 1.0
        gy, gx = np.mgrid[y0:y1, x0:x1]
        yy[k], xx[k] = float((gy * wt).sum()) / tot, float((gx * wt).sum()) / tot
    idx = comp - 1
    miss = np.flatnonzero(~has[1:])                    # 沒找到亮點的塊（平頂整片一樣高）：保留原本的
    ys = np.concatenate([yy, stars.y[miss]])
    xs = np.concatenate([xx, stars.x[miss]])
    idx = np.concatenate([idx, miss])
    return dataclasses.replace(stars, y=ys, x=xs, radius=stars.radius[idx], saturated=stars.saturated[idx])


def keep_stars(stars, keep: np.ndarray):
    """只留 keep 的星（星點遮罩、權重也跟著拿掉被排除的）。"""
    import dataclasses
    lab, n = ndimage.label(stars.mask)
    if n:
        y = np.clip(np.round(stars.y).astype(int), 0, stars.mask.shape[0] - 1)
        x = np.clip(np.round(stars.x).astype(int), 0, stars.mask.shape[1] - 1)
        drop = np.zeros(n + 1, bool)
        drop[lab[y[~keep], x[~keep]]] = True
        drop[lab[y[keep], x[keep]]] = False
        drop[0] = False
        gone = drop[lab]
        mask = stars.mask & ~gone
        weight = np.where(gone, 0, stars.weight).astype(stars.weight.dtype)
    else:
        mask, weight = stars.mask, stars.weight
    return dataclasses.replace(stars, y=stars.y[keep], x=stars.x[keep], radius=stars.radius[keep],
                               saturated=stars.saturated[keep], mask=mask, weight=weight)


def dark_rings(L: np.ndarray, stars, radius: np.ndarray, scale: float) -> np.ndarray:
    """星點旁邊的暗環（疊圖插值、反摺積的振鈴）：遮罩外 3 px 那一圈，8 個方向裡至少 6 個比更外面（5～9 px）
    暗 1.5σ 以上的星，遮罩往外擴 4 px。不然補的時候從暗環往裡面補，每顆星的位置變一個暗洞（LBN576）。
    要整圈都暗：星雲裡的暗帶只在一邊（只看「比背景暗」的話，NGC6888 星雲的暗帶一大片被補掉）。"""
    h, w = L.shape
    mask = np.zeros((h, w), bool)
    n = len(stars.y)
    if n == 0:
        return mask
    px = 1.0 / max(scale, 1e-6)
    noise = ops.pixel_noise(L)
    Ls = ops.blur(L, max(0.7, px))
    th = np.radians(np.arange(0, 360, 45, dtype=np.float32))
    near = np.array([0.5, 1.5, 2.5], np.float32) * px
    far = np.array([5.0, 7.0, 9.0], np.float32) * px
    for a in range(0, n, 4000):
        y = stars.y[a:a + 4000, None, None, None].astype(np.float32)
        x = stars.x[a:a + 4000, None, None, None].astype(np.float32)
        r0 = radius[a:a + 4000, None, None, None].astype(np.float32)
        rr = np.stack([near, far])[None, :, None, :] + r0                       # (n, 2, 1, 3)
        ys = np.clip(y + rr * np.sin(th)[None, None, :, None], 0, h - 1)
        xs = np.clip(x + rr * np.cos(th)[None, None, :, None], 0, w - 1)
        v = ndimage.map_coordinates(Ls, [ys.ravel(), xs.ravel()], order=1).reshape(ys.shape)
        v = v.mean(axis=3)                                                        # (n, 2, 8)
        darker = (v[:, 0] < v[:, 1] - 1.5 * noise).sum(axis=1) >= 6
        for j in np.flatnonzero(darker):
            yi, xi = float(stars.y[a + j]), float(stars.x[a + j])
            r = float(radius[a + j]) + 4 * px
            y0, y1 = int(max(0, yi - r - 1)), int(min(h, yi + r + 2))
            x0, x1 = int(max(0, xi - r - 1)), int(min(w, xi + r + 2))
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= (yy - yi) ** 2 + (xx - xi) ** 2 <= r * r
    return mask


def bright_neighbours(L: np.ndarray, mask: np.ndarray, scale: float, noise_boost: float = 1.0) -> np.ndarray:
    """遮罩附近（12 px 內）亮而緊湊的結構（星點大小）：一起遮住。不然補的時候會把它的亮光抹進補的區域，
    變成一團白斑（例如雙星被找星當成細長的星雲纖維排除，就在旁邊星點的遮罩外面）。大片的亮結構（星雲）不納入。"""
    bg = _background(L)
    hp = L - bg
    sample = hp[:: max(1, L.shape[0] // 512), :: max(1, L.shape[1] // 512)]
    sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) * noise_boost
    bright = hp > 5 * sigma
    lab, n = ndimage.label(bright)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(np.ones_like(L), lab, index=np.arange(1, n + 1))
    # 補洞只用遮罩邊上的像素（調和內插）：貼著遮罩（2 px 內）的才要遮。以前的補法會用到遮罩外約 9 px，
    # M31 一對離遮罩 5 px 的雙星被抹進補的區域，變成一塊灰白色的盾牌形光斑，那時候範圍是 12 px
    near = ndimage.binary_dilation(mask, iterations=int(round(2 / max(scale, 1e-6))) + 1)
    touching = ndimage.maximum(near, lab, index=np.arange(1, n + 1)) > 0
    within = ndimage.minimum(mask, lab, index=np.arange(1, n + 1)) == 0      # 還沒有整個被遮住
    small = sizes <= 300 / max(scale, 1e-6) ** 2
    # 要像星：尖銳（星雲的亮結、纖維片段平緩，不能納入）
    sharp = ndimage.maximum(sharpness(L, scale, hp) * bright, lab, index=np.arange(1, n + 1)) >= sharp_limit(scale)
    cand = np.flatnonzero(touching & within & small & sharp)
    keep = np.zeros(n + 1, bool)
    if len(cand):
        # 而且要圓（星雲纖維上的亮結一邊亮一邊暗，NGC6888 的纖維沿著遮罩一段段被補掉）
        from types import SimpleNamespace
        pos = ndimage.maximum_position(hp, lab, index=cand + 1)
        pts = SimpleNamespace(y=np.array([q[0] for q in pos], float), x=np.array([q[1] for q in pos], float))
        keep[cand + 1] = anisotropy(L, pts, scale) < ANISOTROPY
    return ndimage.binary_dilation(keep[lab], iterations=2)


def smooth_star_colour(stars: np.ndarray, sigma: float) -> np.ndarray:
    """星點層的顏色在 sigma px 範圍內平均（亮度開根號加權），亮度不變。三個色版的星點位置、大小不一樣時
    （紅光焦點不同、大氣色散），亮星會一邊紅一邊青、白色星核外一圈紅；平均後每顆星還是自己的顏色，色環不見。"""
    L = stars.mean(axis=0)
    w = np.sqrt(np.maximum(L, 0)).astype(np.float32)
    den = np.maximum(ops.blur(w, sigma), 1e-12)
    safe = np.maximum(L, 1e-12)
    ratios = np.stack([ops.blur((stars[c] / safe * w).astype(np.float32), sigma) / den for c in range(stars.shape[0])])
    ratios /= np.maximum(ratios.mean(axis=0, keepdims=True), 1e-6)
    return (L[None] * ratios).astype(np.float32)


class StarSplit(Tool):
    kind = "star_split"
    title = T("去星", "Star Removal")
    help = T("分成「無星圖」和「星點層」兩層，之後的步驟可以各自處理（星雲拉深、星點保持小而不爆），"
             "最後用「合成星點」加回去。建議在拉伸前、反摺積之後做。",
             "Splits the image into starless and stars layers so they can be processed separately; "
             "“Recombine Stars” puts them back. Best done before stretching, after deconvolution.")
    phase = "linear"
    icon = "stars"
    maskable = False
    auto_layer = "main"

    def defaults(self) -> dict:
        return {"threshold": 5.0, "grow": 2.0, "texture": True, "spikes": True, "fringes": True}

    def controls(self) -> list[dict]:
        return [
            slider("threshold", T("偵測門檻", "Detection threshold"), 2, 12, 0.5, "{:.1f} σ",
                   help=T("數字小會抓到更暗的星，但也可能把星雲的亮點當成星。", "Lower finds fainter stars but may catch "
                                                                    "nebula knots.")),
            slider("grow", T("範圍外擴", "Grow"), 0, 12, 0.5, "{:.1f} px",
                   help=T("無星圖上留下星暈的話調大。", "Increase if halos remain on the starless image.")),
            toggle("texture", T("補上噪聲紋理", "Match noise texture")),
            toggle("fringes", T("修正星點色邊", "Fix star colour fringes"),
                   help=T("三色的星點位置、大小不一樣（色差、大氣色散）時，亮星會一邊紅一邊青或外圍一圈紅；"
                          "把星點的顏色在小範圍內平均，每顆星保留自己的顏色。",
                          "When the colour channels' stars differ in size or position (chromatic aberration, "
                          "atmospheric dispersion), bright stars get red/cyan fringes; this evens out each star's "
                          "colour while keeping it.")),
            toggle("spikes", T("一起去掉星暈和繞射芒", "Remove halos and spikes"),
                   help=T("每顆亮星量出星暈的大小和繞射芒的長度，一起補掉（折射鏡沒有繞射芒時自動略過）。",
                          "Measures each bright star's halo and diffraction spikes and removes them too (spikes "
                          "are skipped automatically when there are none).")),
        ]

    def run(self, state: State, params: dict, layer: str, mask: dict | None, ctx: Ctx) -> State:
        out = state.copy()
        if state.separated:
            out.warnings.append(T("已經分離過星點了", "Stars are already separated"))
            return out
        ctx.layer = "main"
        img = ops.planes(state.layers["main"])
        L = img.mean(axis=0)
        # 前面有 AI 降噪的話噪聲變小，門檻乘回一部分（開根號）：整個乘回去（跟不用 AI 時一樣）的話，
        # AI 後看得清楚的暗星沒抓到、留在無星圖上，拉伸後一顆顆很明顯（LBN576）；完全不乘又會抓到 AI 的紋理
        boost = float(state.notes.get("main", {}).get("detail_noise_ratio", 1.0))
        stars = pdetail.find_stars(L, float(params.get("threshold", 5.0)) * np.sqrt(max(boost, 1.0)))
        stars = recenter_stars(L, stars, ctx.scale)
        # 星雲的亮結也會被找星抓到：不夠尖銳的不當星（留在無星圖）
        like = star_like(L, stars, ctx.scale)
        ctx.view["not_stars"] = int((~like).sum())
        stars = keep_stars(stars, like)
        ctx.check()
        mask, core_r = star_footprints(L, stars, ctx.scale, float(params.get("grow", 2.0)))
        halos = None
        if params.get("spikes", True):
            ratio = state.notes.get("main", {}).get("noise_ratio")
            provider = ctx.meta.get("spike_angle")
            given = provider() if provider is not None else NOT_GIVEN
            if given is not NOT_GIVEN and given is not None:
                for op, _cin in ctx.frame_ops or []:          # 翻轉時角度反過來；轉 90° 不影響（四重對稱）
                    if op[0] in ("flip_h", "flip_v"):
                        given = (-given) % 90
            extra, angle, halos = halo_spike_mask(L, stars, ctx.scale, float(ratio[0]) if ratio else 1.0, core_r,
                                                  given, img)
            if extra is not None:
                mask |= extra
            ctx.view["spike_angle"] = angle
        ratio = state.notes.get("main", {}).get("noise_ratio")
        mask |= bright_neighbours(L, mask, ctx.scale, float(ratio[0]) if ratio else 1.0)
        mask |= dark_rings(L, stars, core_r + float(params.get("grow", 2.0)) / ctx.scale, ctx.scale)
        ctx.progress(0.4, "inpaint")
        if halos is not None:
            # 星暈只扣超過 1 倍噪聲的部分（r²／(r＋σ)），無星圖扣多少、星點層就放多少：星暈外圍量到噪聲裡
            # （AI 降噪後量得很遠），整個扣的話中等亮度的星周圍各有一圈淡淡的光盤；只在星點層打折的話，
            # 淡的星暈兩邊都沒了，合成後每顆星的位置一個暗洞
            for c in range(img.shape[0]):
                n = ops.pixel_noise(img[c])
                halos[c] = halos[c] * halos[c] / (halos[c] + n + 1e-12)
        smooth = backend.harmonic_fill(img - halos if halos is not None else img, mask, ctx.check)
        ctx.check()
        # 星點層＝原圖 − 補好的背景，但先扣掉 1 倍噪聲再去負值：直接去負值的話正的噪聲留下、負的切掉，
        # 每顆星周圍多一個帶噪聲的亮盤（遮罩範圍），合成後變成一圈光暈。遮罩邊緣漸變
        # 遮罩邊緣往內 4 px 漸變、噪聲門檻用平滑的曲線（r²／(r＋σ)，大的值≈r−σ），不然星點層拉伸後
        # 光暈在遮罩邊緣突然切掉，星點外圍看起來有一圈邊
        resid = img - smooth
        dist = ndimage.distance_transform_edt(mask)
        edge = ops.smoothstep(0.0, 4.0 / ctx.scale, dist).astype(np.float32)
        layers = []
        for c in range(img.shape[0]):
            r = np.maximum(resid[c], 0)
            soft = r * r / (r + ops.pixel_noise(img[c]) + 1e-12) * edge
            if halos is not None:
                soft += halos[c] * (1.0 - edge)
            layers.append(soft)
        stars_layer = np.stack(layers).astype(np.float32)
        if img.shape[0] >= 3 and params.get("fringes", True):
            stars_layer = smooth_star_colour(stars_layer, 2.0 / ctx.scale)
        starless = smooth
        if params.get("texture", True):
            # 噪聲紋理：整張天空量一個噪聲（星點以外的高頻），亮的地方照光子噪聲（∝√亮度）放大；
            # 三色用同一份紋理（去馬賽克後的噪聲本來就三色相關），按各色亮度比例分配
            rng = np.random.default_rng(12345)
            hp = (L - ops.blur(L, 1.0))[~mask]
            _m, sigma0 = ops.robust_stats(hp)
            sky, _s = ops.robust_stats(L[~mask] if ctx.ref is None else ctx.whole(img).mean(axis=0))
            level = ops.blur(smooth.mean(axis=0), 3.0)
            ref = max(abs(sky), 1e-6)
            sigma = sigma0 * 1.15 * np.sqrt(np.clip(np.maximum(level, ref) / ref, 1.0, 16.0))
            grain = rng.standard_normal(L.shape).astype(np.float32) * sigma * ops.blur(mask.astype(np.float32), 0.7)
            Ls = np.maximum(smooth.mean(axis=0), 1e-9)
            for c in range(img.shape[0]):
                share = np.clip(smooth[c] / Ls, 0.3, 3.0) if img.shape[0] > 1 else 1.0
                starless[c] = starless[c] + grain * share
        out.layers = {"starless": starless.astype(np.float32), "stars": stars_layer}
        ph = state.phase.get("main", "linear")
        out.phase = {"starless": ph, "stars": ph}
        main_notes = state.notes.get("main", {})
        out.notes = {"starless": dict(main_notes), "stars": dict(main_notes)}
        ctx.view["stars_found"] = len(stars.y)
        return out


register(StarSplit())


class StarMerge(Tool):
    kind = "star_merge"
    title = T("合成星點", "Recombine Stars")
    help = T("把星點層加回無星圖。可以調星點的亮度、大小與顏色。星點層還沒拉伸的話會自動用保色拉伸。",
             "Puts the stars back on the starless image, with control over star brightness, size and colour. "
             "A linear stars layer is stretched automatically (colour-preserving).")
    phase = "nonlinear"
    icon = "stars"
    maskable = False

    def defaults(self) -> dict:
        return {"mode": "screen", "brightness": 1.0, "reduce": 0.0, "saturation": 1.2, "demagenta": 0.0, "beta": 60.0}

    def controls(self) -> list[dict]:
        return [
            choice("mode", T("合成方式", "Blend"), [("screen", T("濾色（不會過曝）", "Screen (no clipping)")),
                                                  ("add", T("相加", "Add"))]),
            slider("brightness", T("星點亮度", "Star brightness"), 0, 2, 0.05, "{:.0%}"),
            slider("reduce", T("縮星", "Shrink stars"), 0, 1, 0.05, "{:.0%}"),
            slider("saturation", T("星點飽和度", "Star saturation"), 0, 2.5, 0.05, "{:.0%}"),
            slider("demagenta", T("去紫", "Remove magenta"), 0, 1, 0.05, "{:.0%}",
                   help=T("窄帶合成（SHO、HOO）的星點常常偏紫，往中性拉回。", "Narrowband stars often turn magenta; "
                                                                      "this pulls them back toward neutral.")),
        ]

    def run(self, state: State, params: dict, layer: str, mask: dict | None, ctx: Ctx) -> State:
        out = state.copy()
        if not state.separated:
            out.warnings.append(T("還沒有去星，沒有星點層可以合成", "There is no stars layer to recombine"))
            return out
        base = ops.planes(state.layers["starless"])
        stars = ops.planes(state.layers["stars"])
        if state.phase.get("starless") == "linear":
            out.warnings.append(T("無星圖還沒拉伸：先用自動拉伸顯示", "The starless layer is still linear; it was "
                                                             "auto-stretched"))
            ctx.layer = "starless"
            base = ops.auto_display(base, ctx.whole(base) if ctx.ref is not None else None)
        if state.phase.get("stars") == "linear":
            black = [0.0] * len(stars)
            stars = ops.arcsinh_stretch(stars, float(params.get("beta", 60.0)) * 50, black)
        stars = np.clip(stars, 0, 1)
        r = float(params.get("reduce", 0.0))
        if r > 0:
            small = np.stack([ndimage.grey_erosion(s, size=(3, 3)) for s in stars])
            stars = stars + (small - stars) * r
            stars *= 1 - 0.4 * r
        dm = float(params.get("demagenta", 0.0))
        if stars.shape[0] >= 3 and dm > 0:
            excess = np.maximum(np.minimum(stars[0], stars[2]) - stars[1], 0)
            stars = stars.copy()
            stars[0] -= excess * dm
            stars[2] -= excess * dm
        sat = float(params.get("saturation", 1.2))
        if stars.shape[0] >= 3 and abs(sat - 1) > 1e-3:
            Y = ops.luminance(stars)
            stars = np.clip(Y[None] + (stars - Y[None]) * sat, 0, 1)
        stars = stars * float(params.get("brightness", 1.0))
        if params.get("mode", "screen") == "add":
            merged = base + stars
        else:
            merged = 1 - (1 - np.clip(base, 0, 1)) * (1 - np.clip(stars, 0, 1))
        out.layers = {"main": np.clip(merged, 0, 1).astype(np.float32)}
        out.phase = {"main": "nonlinear"}
        out.notes = {"main": dict(state.notes.get("starless", {}))}
        return out


register(StarMerge())
