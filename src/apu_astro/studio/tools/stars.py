"""去星（分成無星圖＋星點層）與合成星點。

去星（傳統方法，之後可以換成 AI 模型）：找星點 → 星點範圍用周圍的值補（由小到大的正規化卷積）→ 補的地方加上
跟附近一樣的噪聲紋理（不然無星圖上會有一個個太光滑的圓斑）→ 星點層＝原圖 − 無星圖。
在線性階段做，之後兩層分開拉伸，最後合成星點時加回去。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from apu_processing import detail as pdetail
from scipy import ndimage

from .. import backend, ops
from . import T, Ctx, State, Tool, choice, register, slider, toggle


SMOOTH = 7      # 沿方向平均的長度（px）：繞射芒、星暈外圍每個像素都比噪聲暗，要平均才量得到
# 繞射芒：沿線的亮度降到 SPIKE_LIMIT 倍（沿線取樣的）噪聲為止，遮到量到長度的 SPIKE_EXTRA 倍。以前 3 倍、1.3 倍：
# 尾巴每個像素不到 1σ，但一條長線強拉伸後還是看得到（LBN576 亮星的繞射芒留下 150 px 的淡線）
SPIKE_LIMIT = 2.0
SPIKE_EXTRA = 1.8


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


def _label_max(values: np.ndarray, lab: np.ndarray, n: int, where: np.ndarray | None = None) -> np.ndarray:
    """每個連通區（1..n）的最大值，(n+1,)，第 0 個是背景。ndimage.maximum 對幾萬個區塊很慢（逐塊挑）。"""
    sel = lab > 0 if where is None else (lab > 0) & where
    out = np.full(n + 1, -np.inf, np.float64)
    np.maximum.at(out, lab[sel], values[sel])
    return out


def _background(L: np.ndarray, f: int = 8) -> np.ndarray:
    """不受星點影響的背景：f×f 區塊取中位數 → 再取 7×7 中位數 → 放大回原尺寸。
    （用高斯模糊當背景的話，亮星附近的背景會被星點拉高，星暈、繞射芒反而變成負的）"""
    h, w = L.shape
    h2, w2 = max(f, h // f * f), max(f, w // f * f)
    if h < f or w < f:
        return np.full_like(L, float(np.median(L)))
    blocks = L[:h2, :w2].reshape(h2 // f, f, w2 // f, f).transpose(0, 2, 1, 3).reshape(h2 // f, w2 // f, -1)
    small = ndimage.median_filter(np.median(blocks, axis=2), size=7, mode="nearest").astype(np.float32)
    # 放大回原尺寸：雙線性，分兩個方向做（整張的座標網格 map_coordinates 一次要 2 個 160 MB 的陣列）
    def axis_weights(n, m):
        z = np.clip((np.arange(n) + 0.5) / f - 0.5, 0, m - 1)
        i0 = np.minimum(np.floor(z).astype(int), m - 1)
        return i0, np.minimum(i0 + 1, m - 1), (z - i0).astype(np.float32)
    y0, y1, fy = axis_weights(h, small.shape[0])
    x0, x1, fx = axis_weights(w, small.shape[1])
    rows = small[y0] * (1 - fy)[:, None] + small[y1] * fy[:, None]
    return (rows[:, x0] * (1 - fx)[None] + rows[:, x1] * fx[None]).astype(np.float32)


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
    spike_limit = SPIKE_LIMIT * mad_of(singles) * noise_boost

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
            tt = np.arange(0, length * SPIKE_EXTRA + SMOOTH, 0.5)
            for off in np.arange(-width - 1.5, width + 1.51, 0.5):
                ys = np.round(cy + tt * np.sin(a) + off * np.cos(a)).astype(int)
                xs = np.round(cx + tt * np.cos(a) - off * np.sin(a)).astype(int)
                ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                mask[ys[ok], xs[ok]] = True
    return (mask if mask.any() else None), angle, halos


def detect_stars(L: np.ndarray, threshold: float):
    """找星：8 px 以下的細結構（starlet 前三層）比「細結構真正的噪聲」亮 threshold 倍的連通區，每一塊一顆
    （重心用細結構加權），跟 pdetail.find_stars 一樣，差在三個地方：
    1. 噪聲：find_stars 用細結構整體的 MAD，降噪後背景的噪聲很小、細結構裡星點反而佔大部分，量到的「噪聲」被星點
       撐大好幾倍（LBN576 AI 降噪後：設 9.4σ，峰值 10～20σ 的星只找到 2%，無星圖上暗星幾乎全留著）。這裡反覆剔掉
       3σ 以外的值（星點），剩下的才是噪聲；AI 留下的紋理也算在噪聲裡，不會被當成星
    2. 細長的塊不丟：門檻低了以後靠很近的星連成一條（雙星、星串），find_stars 當成星雲纖維整塊丟掉
       （LBN576 一顆 300σ 的星就這樣留在無星圖上）。之後 recenter_stars 拆成一顆顆，星雲纖維由 star_like 擋
    3. mask 是星核的連通區（沒有外擴）：recenter_stars 靠它對應每顆星。find_stars 的 mask 外擴過，相鄰的星連在一起、
       塊數對不上，重新定位整個被跳過（LBN576 2241 顆星只有 1022 塊）"""
    from apu_processing import denoise as pdenoise
    _coeffs, smooth = pdenoise.starlet(L, 3)
    top = L - smooth
    h, w = L.shape
    v = top[:: max(1, h // 512), :: max(1, w // 512)].ravel().astype(np.float64)
    sig = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    for _ in range(8):
        m = float(np.median(v))
        sig = 1.4826 * float(np.median(np.abs(v - m)))
        keep = np.abs(v - m) < 3.0 * sig
        if keep.all():
            break
        v = v[keep]
    cand = top > threshold * max(sig, 1e-12)
    lab, n = ndimage.label(cand)
    if n == 0:
        z = np.zeros(0)
        return pdetail.Stars(z, z, z, np.zeros(0, bool), np.zeros((h, w), bool), np.zeros((h, w), np.float32), sig)
    py, px = np.nonzero(cand)
    li = lab[py, px]
    area = np.bincount(li, minlength=n + 1)[1:]
    ok = area >= 3
    wts = np.maximum(top[py, px], 0).astype(np.float64)
    tot = np.maximum(np.bincount(li, wts, minlength=n + 1)[1:], 1e-30)
    ys = np.bincount(li, wts * py, minlength=n + 1)[1:] / tot
    xs = np.bincount(li, wts * px, minlength=n + 1)[1:] / tot
    # 飽和：塊裡的最大值接近整張的最大值，而且至少 5 個像素一樣高
    vals = L[py, px]
    peak = np.full(n + 1, -np.inf)
    np.maximum.at(peak, li, vals)
    flat = np.bincount(li, vals >= 0.995 * peak[li], minlength=n + 1)[1:]
    peak = peak[1:]
    sat = (peak >= 0.97 * float(L.max())) & (flat >= 5)
    keep_lab = np.concatenate([[False], ok])
    mask = keep_lab[lab]
    return pdetail.Stars(ys[ok], xs[ok], np.sqrt(area[ok] / np.pi), sat[ok], mask, mask.astype(np.float32), sig)


def _discs(shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """每個 (y, x) 畫一個半徑 r 的圓（小數位置、小數半徑）。"""
    h, w = shape
    mask = np.zeros((h, w), bool)
    for y, x, r in zip(ys, xs, radii):
        ri = int(np.ceil(r)) + 1
        yi, xi = int(round(y)), int(round(x))
        y0, y1, x0, x1 = max(0, yi - ri), min(h, yi + ri + 1), max(0, xi - ri), min(w, xi + ri + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return mask


def star_footprints(L: np.ndarray, stars, scale: float, grow: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """每顆星的星核範圍（AI 降噪的去斑用：只填星核，見 ai._coarse_pass）：從星心往 8 個方向取亮度剖面（取第二暗的
    方向），扣掉外圈的局部背景，降到峰值 10% 的半徑 × 1.6＋1 px（但不超過半高半徑 × 2.5＋1.5、也不超過降到噪聲
    以下的半徑＋1.5）再加 grow 畫成圓（px 都是完整尺寸）。回傳 (遮罩, 每顆星的半徑，不含 grow)。
    去星不用這個（亮星外圍還有好幾 σ，見 star_model）。
    不用找星的連通區域：亮星常跟旁邊的星雲連成一塊，整塊被當成星（NGC6888 星雲亮邊被搬進星點層）。
    半徑、圓心都用小數：快速預覽的一個像素是完整尺寸的 4 px，取整數的話每顆星大一圈，預覽補掉的比成品多。"""
    h, w = L.shape
    n = len(stars.y)
    if n == 0:
        return np.zeros((h, w), bool), np.zeros(0)
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
    return _discs((h, w), stars.y, stars.x, R + grow * px), R


FOOT_LEVEL = 0.3      # 星光降到幾倍像素噪聲（完整解析度）算到底
HOLE_LEVEL = 3.0      # 星點模型還有幾倍像素噪聲以上的地方挖掉補（模型不夠準），以下扣模型
FOOT_RMAX = 40.0      # 剖面量到幾 px（完整尺寸）；更外面的星暈用扣的（halo_spike_mask）
OWN_RMAX = 120.0      # 飽和、特別亮的星自己的剖面量到幾 px（完整尺寸）
OWN_PEAK = 1000.0     # 快速預覽上峰值幾倍噪聲以上的星量自己的剖面（不用疊的形狀；見 star_model）
FOOT_REF_MIN = 12     # 量標準星點形狀至少要幾顆星


def _ring_samples(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """每顆星在半徑 rr 上 16 個方向的值（每個半徑由暗到亮排好），(n, 16, len(rr))。"""
    h, w = L.shape
    th = np.radians(np.arange(0, 360, 22.5, dtype=np.float32))
    out = np.empty((len(ys), 16, len(rr)), np.float32)
    for a in range(0, len(ys), 2000):                # 一次 2000 顆，省記憶體
        y = ys[a:a + 2000, None, None].astype(np.float32)
        x = xs[a:a + 2000, None, None].astype(np.float32)
        py = np.clip(y + rr[None, None, :] * np.sin(th)[None, :, None], 0, h - 1)
        px = np.clip(x + rr[None, None, :] * np.cos(th)[None, :, None], 0, w - 1)
        out[a:a + 2000] = np.sort(ndimage.map_coordinates(L, [py.ravel(), px.ravel()], order=1).reshape(py.shape),
                                  axis=1)
    return out


def _radii_grid(scale: float, rmax: float) -> np.ndarray:
    """剖面的半徑：每完整尺寸 1 px 一點。"""
    step = 1.0 / max(1, int(round(scale)))
    return np.arange(0, rmax + 1e-6, step, dtype=np.float32)


def _centre_bg(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, rr: np.ndarray):
    """每顆星的星心值、外圈（rr 的 80% 以外）的中位數與第 1/4 暗的值。"""
    far = rr[rr >= 0.8 * rr[-1]]
    smp = _ring_samples(L, ys, xs, np.concatenate([[0.0], far]).astype(np.float32))
    ring = smp[:, :, 1:].reshape(len(ys), -1)
    return smp[:, 0, 0], np.median(ring, axis=1), np.percentile(ring, 25, axis=1)


def _profiles(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, rr: np.ndarray):
    """每顆星 16 個方向的亮度剖面：(各方向的中位數 (n, r), 第 4 暗的方向 (n, r))。"""
    smp = _ring_samples(L, ys, xs, rr)
    return 0.5 * (smp[:, 7] + smp[:, 8]), smp[:, 3]


def _crossing(s: np.ndarray, rr: np.ndarray, level) -> np.ndarray:
    """剖面 (n, r)（先取累積最小值）第一次降到 level 以下的半徑（線性內插到小數）；一直沒降到＝最後一點。"""
    s = np.minimum.accumulate(s, axis=1)
    level = np.broadcast_to(np.asarray(level, np.float32), (len(s),))
    below = s < level[:, None]
    j = np.where(below.any(axis=1), below.argmax(axis=1), len(rr) - 1)
    jm = np.maximum(j - 1, 0)
    i = np.arange(len(s))
    a, b = s[i, jm], s[i, j]
    f = np.clip((a - level) / np.maximum(a - b, 1e-30), 0, 1)
    return np.where(j > 0, rr[jm] + f * (rr[j] - rr[jm]), rr[0])


def _star_shape(rel: np.ndarray) -> np.ndarray:
    """一群亮度差不多的星的剖面 (k, r)（扣過背景）→ 星點形狀（中心＝1）：中位數剖面 ÷ 中位數峰值（各自除的話
    某個色版很暗的星會除以接近 0 的數），沿半徑平滑（外圍很淡，平均的長度隨距離變長），只准往外變暗、不小於 0。"""
    g = np.median(rel, axis=0) / max(float(np.median(rel[:, 0])), 1e-30)
    cs = np.concatenate([[0.0], np.cumsum(g)])
    t = np.arange(len(g))
    half = np.maximum(0, np.round(0.125 * t)).astype(int)
    lo, hi = np.clip(t - half, 0, len(g) - 1), np.clip(t + half + 1, 1, len(g))
    g = (cs[hi] - cs[lo]) / (hi - lo)
    g[0] = 1.0
    return np.clip(np.minimum.accumulate(g), 0, 1).astype(np.float32)


@dataclass
class StarModel:
    rr: np.ndarray        # 剖面的半徑（這個解析度的像素）
    shapes: np.ndarray    # (B, C, r) 各亮度組、各色版的星點形狀（中心＝1）
    pos: np.ndarray       # (n,) 每顆星在亮度組之間的位置（小數：1.3＝第 1 組和第 2 組 7:3 內插）
    amp: np.ndarray       # (n, C) 每顆星每色版中心比背景亮多少；不用模型的星（飽和）是 0
    hole: np.ndarray      # (n,) 挖掉補的半徑（不含外擴）
    reach: np.ndarray     # (n,) 星光降到 FOOT_LEVEL 倍噪聲的半徑
    own_idx: np.ndarray | None = None    # 量自己剖面的星（飽和、特別亮的）
    own_rr: np.ndarray | None = None     # 它們剖面的半徑
    own_prof: np.ndarray | None = None   # (k, C, r) 它們自己的剖面（扣過背景，絕對值）

    def shape_of(self, i: int) -> np.ndarray:
        """第 i 顆星的形狀 (C, r)。"""
        b = int(np.floor(self.pos[i]))
        f = float(self.pos[i] - b)
        if b + 1 >= len(self.shapes):
            return self.shapes[-1]
        return (1 - f) * self.shapes[b] + f * self.shapes[b + 1]


def star_model(planes: np.ndarray, stars, scale: float, hp: np.ndarray | None = None,
               flat: np.ndarray | None = None) -> StarModel:
    """每顆星的星點模型與範圍。
    同一張圖亮度差不多的星形狀都差不多：每 2 倍亮度一組（至少 FOOT_REF_MIN 顆）疊出中位數形狀，
    每顆星＝自己的峰值 × 自己亮度的形狀（組之間內插；每個色版各自量：藍星的星暈比較大）。不同亮度的形狀不一樣：
    暗星相對胖（AI 降噪、位置誤差讓峰值偏低），亮星的星暈相對大（LBN576：12 px 處 2000σ 的星是峰值的 0.1%、
    250σ 的 0.04%）；全部用同一個形狀的話暗星和最亮的星外圍都扣不夠，洞的邊上還有星光、補出來偏亮。
    - 星核（模型 HOLE_LEVEL 倍噪聲以上）挖掉補：這裡模型差一點就是好幾 σ（反摺積、形狀隨位置變）
    - 外圍扣模型：星雲的紋路還在。以前整顆挖掉補，範圍小的話邊上還有星光（LBN576 200～2000σ 的星範圍只到 6 px、
      邊上 2σ，補出來整個洞偏亮，星心殘影 1.5σ）；範圍放大到星光降到 0.3σ 的話密集星場一半的面積被補平，
      星雲殼的纖維變成一片平滑的光（NGC6888）
    - 單顆星在 0.3σ 的地方量不準（噪聲、旁邊的星、星雲），以前每顆自己量、暗星常常量得很大（NGC6888 遮罩
      蓋掉半張圖），只好加上限；用疊出來的形狀就沒有這個問題
    飽和的（快速預覽上再加很亮的，見下面的 own）量自己的剖面（第 4 暗的方向，避開星雲纖維、旁邊的星），扣自己的剖面；
    挖掉補的只有各方向明顯不一樣的星核；剖面不再下降（星雲的平台）就停。
    噪聲：快速預覽時 AI 降噪後的噪聲是相關的，縮小 4 倍噪聲只小 1.6～2.9 倍，用縮圖的噪聲 × √倍率 估完整解析度的。"""
    C = planes.shape[0]
    n = len(stars.y)
    px = 1.0 / max(scale, 1e-6)
    L = planes.mean(axis=0)
    rr = _radii_grid(scale, FOOT_RMAX * px)
    empty = StarModel(rr, np.zeros((0, C, len(rr)), np.float32), np.zeros(n), np.zeros((n, C), np.float32),
                      np.full(n, 2.0 * px, np.float32), np.full(n, 2.0 * px, np.float32))
    if n == 0:
        return empty
    ys, xs = np.asarray(stars.y, np.float64), np.asarray(stars.x, np.float64)
    boost = np.sqrt(scale) if scale > 1.5 else 1.0
    noise = ops.pixel_noise(L) * boost
    centre, bg, bgq = _centre_bg(L, ys, xs, rr)
    peak = (centre - bg) / max(noise, 1e-30)
    if flat is None:
        flat = saturated_tops(L - _background(L) if hp is None else hp, scale)
    y = np.clip(np.round(ys).astype(int), 0, L.shape[0] - 1)
    x = np.clip(np.round(xs).astype(int), 0, L.shape[1] - 1)
    sat = flat[y, x] | np.asarray(stars.saturated, bool)
    lp = np.log2(np.maximum(peak, 1.0))
    chans = [L] if C == 1 else list(planes)
    cbg = [(centre, bg)] if C == 1 else [_centre_bg(p, ys, xs, rr)[:2] for p in chans]
    amp = np.maximum(np.stack([c - b for c, b in cbg], axis=1), 0).astype(np.float32)
    # 亮度分組；形狀只用每組最多 200 顆量（星很多的時候省時間，中位數已經很穩）
    rng = np.random.default_rng(0)
    groups = []
    if (~sat).any():
        top_lp = lp[~sat].max() if scale <= 1.5 else min(lp[~sat].max(), np.log2(OWN_PEAK))   # 見下面的 own
        for c0 in np.arange(np.floor(lp[~sat].min()), top_lp + 0.5, 0.5):
            sel = np.flatnonzero(~sat & (np.abs(lp - c0) < 0.5))
            if len(sel) < FOOT_REF_MIN:
                continue
            groups.append((c0, rng.choice(sel, min(len(sel), 200), replace=False)))
    if not groups:
        model = empty
        own = np.ones(n, bool)
    else:
        sub = np.unique(np.concatenate([g for _c, g in groups]))
        where = np.full(n, -1)
        where[sub] = np.arange(len(sub))
        rel_l = _profiles(L, ys[sub], xs[sub], rr)[0] - bg[sub, None]
        rel_c = [rel_l] if C == 1 else [_profiles(p, ys[sub], xs[sub], rr)[0] - b[sub, None]
                                        for p, (_c, b) in zip(chans, cbg)]
        centres = [c0 for c0, _g in groups]
        sl = np.stack([_star_shape(rel_l[where[g]]) for _c, g in groups])
        shapes = np.stack([np.stack([_star_shape(rc[where[g]]) for rc in rel_c]) for _c, g in groups])
        pos = np.interp(lp, centres, np.arange(len(centres)))
        b = np.floor(pos).astype(int)
        f = (pos - b)[:, None]
        gl = (1 - f) * sl[b] + f * sl[np.minimum(b + 1, len(sl) - 1)]       # (n, r) 每顆星的亮度形狀
        prof = np.maximum(peak, 0)[:, None] * gl                              # 單位：噪聲
        # 快速預覽：很亮的（OWN_PEAK 以上）、比最亮的那組再亮 2 倍以上的也量自己的。縮小 4 倍後飽和星的平頂被平均掉、
        # 不算飽和，用最亮那組的形狀扣的話 16～22 px 少扣 3σ，洞整個是一塊亮盤（完整解析度上這些星是飽和的，本來就量自己的）。
        # 完整解析度不這樣做：沒飽和的亮星用疊的形狀比較準（分組一直分到最亮的星；自己的剖面在「盡頭」切掉、扣掉盡頭外的底，
        # 盡頭外的星暈還有好幾 σ：LBN576 一顆 14000σ 的星盡頭 19 px、20 px 處還有 4σ，2000σ 以上的星洞裡平均 +0.5～0.9σ，
        # 疊的形狀 0.1～0.3σ；NGC6888 殼上一顆 3800σ 的星盡頭被星雲撐到 56 px）
        own = sat.copy()
        if scale > 1.5:
            own |= (peak > OWN_PEAK) | (lp > centres[-1] + 1.0)
        amp[own] = 0
        model = StarModel(rr, shapes.astype(np.float32), pos, amp,
                          _crossing(prof, rr, HOLE_LEVEL).astype(np.float32),
                          _crossing(prof, rr, FOOT_LEVEL).astype(np.float32))
    hole, reach = model.hole, model.reach
    top = float(hole[~own].max()) if (~own).any() else 2.0 * px
    hole = np.clip(hole, 1.5 * px, FOOT_RMAX * px)
    reach = np.clip(np.maximum(reach, hole), 1.5 * px, FOOT_RMAX * px)
    if own.any():
        oi = np.flatnonzero(own)
        ro = _radii_grid(scale, OWN_RMAX * px)
        smp = _ring_samples(L, ys[oi], xs[oi], ro)
        _c, _b, bq = _centre_bg(L, ys[oi], xs[oi], ro)
        s = _smooth_radial((smp[:, 3] - bq[:, None]) / max(noise, 1e-30))   # 第 4 暗的方向（避開星雲纖維、旁邊的星）
        # 剖面到哪裡為止：往外 20 px 降不到現在高度（比最外面高多少）的 15%、或降不到 0.15σ。真的星暈越往外越暗得快
        # （冪次），星雲是平台。只看淡的地方（9σ 以下）：飽和星的平頂也不下降。只看相鄰幾 px 的話太敏感，
        # 星暈還有 5σ 就停了，整條剖面又扣掉停的地方的值，等於少扣 5σ（快速預覽上亮星一塊亮盤）
        gap = max(1, int(round(20 * px / max(float(ro[1] - ro[0]), 1e-6))))
        tail = np.median(s[:, -gap:], axis=1)
        drop = s[:, :-gap] - s[:, gap:]
        height = s[:, :-gap] - tail[:, None]
        flat = ((drop < 0.15 * height) | (drop < 0.5 * FOOT_LEVEL)) & (s[:, :-gap] < 30 * FOOT_LEVEL)
        # 各方向的差異（第 4 暗和第 4 亮的方向差，扣掉噪聲本身的差）
        spread = (smp[:, 12] - smp[:, 3]) / 1.35 / max(noise, 1e-30)
        excess = _smooth_radial(np.sqrt(np.maximum(spread * spread - 1.0, 0)))
        # 挖掉補的範圍：最外面一個「各方向還明顯不一樣、星光還有 HOLE_LEVEL 以上」的半徑（反摺積、繞射芒、色差）。
        # 不能用第一次一致的地方：飽和的平頂各方向一樣高。更外面的星暈各方向一樣，扣自己的剖面就乾淨
        # （以前挖到星光降到 3σ：一顆 37000σ 的星要挖 80 px）
        # 「星暈」比各方向的差異還小＝其實是星雲（弧形的星雲取各方向中位數也是越往外越暗），停在那裡。先決定到哪裡停，
        # 洞再限制在那之內：星雲上各方向本來就不一樣、也比遠處亮 3σ 以上，不限制的話洞一路擴到 120 px
        # （NGC6888 殼上的亮星，整片纖維被抹平）
        flat |= (height < excess[:, :-gap]) & (s[:, :-gap] < 30 * FOOT_LEVEL)
        flat = np.concatenate([flat, np.ones((len(s), gap), bool)], axis=1)
        j_stop = flat.argmax(axis=1)
        stop = ro[j_stop]
        uneven = (excess >= 1.0) & (s >= HOLE_LEVEL) & (np.arange(len(ro))[None, :] <= j_stop[:, None])
        last = np.where(uneven.any(axis=1), len(ro) - 1 - np.argmax(uneven[:, ::-1], axis=1), 0)
        after = (np.arange(len(ro))[None, :] >= j_stop[:, None]) & (np.arange(len(ro))[None, :] < j_stop[:, None] + gap)
        # 至少跟用形狀算的一樣大（沒飽和的亮星：只挖各方向不一樣的地方的話洞比較小）
        own_h = np.minimum(ro[last] + ro[1] - ro[0], stop)
        hole[oi] = np.clip(np.maximum(own_h, hole[oi]), top, OWN_RMAX * px)
        reach[oi] = np.maximum(np.minimum(_crossing(s, ro, FOOT_LEVEL), stop), hole[oi])
        # 模型：每色版自己的剖面（各方向的中位數；不對稱的部分在挖掉的範圍裡），平滑、只准往外變暗，
        # 扣掉盡頭外面那一段的中位數當底（不扣星雲的平台），盡頭以外是 0
        profs = []
        for P in chans:
            _cc, bgc, _bq = _centre_bg(P, ys[oi], xs[oi], ro)
            lo = _smooth_radial(_profiles(P, ys[oi], xs[oi], ro)[0] - bgc[:, None])
            floor = np.array([np.median(lo[i, after[i]]) for i in range(len(oi))])
            lo = np.clip(np.minimum.accumulate(lo, axis=1) - floor[:, None], 0, None)
            lo[ro[None, :] > stop[:, None]] = 0
            profs.append(lo)
        model.own_idx, model.own_rr, model.own_prof = oi, ro, np.stack(profs, axis=1).astype(np.float32)
    model.hole, model.reach = hole, reach
    return model


def _smooth_radial(p: np.ndarray) -> np.ndarray:
    """剖面 (n, r) 沿半徑平滑，平均的長度隨距離變長（距離的 1/4；外圍很淡要平均才量得到，靠近星核不能平均）。"""
    cs = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)
    t = np.arange(p.shape[1])
    half = np.maximum(0, np.round(0.125 * t)).astype(int)
    lo, hi = np.clip(t - half, 0, p.shape[1] - 1), np.clip(t + half + 1, 1, p.shape[1])
    return ((cs[:, hi] - cs[:, lo]) / (hi - lo)).astype(np.float32)


def render_stars(model: StarModel, stars, shape: tuple[int, int], floor: np.ndarray) -> np.ndarray:
    """星點模型畫成影像 (C, H, W)：每顆星畫到模型降到 floor（每色版，通常是噪聲的 5%）為止。"""
    C = model.amp.shape[1]
    h, w = shape
    out = np.zeros((C, h, w), np.float32)
    rr = model.rr
    if model.own_idx is not None:                   # 飽和、特別亮的星：自己的剖面
        ro = model.own_rr
        so = float(ro[1] - ro[0]) if len(ro) > 1 else 1.0
        for j, i in enumerate(model.own_idx):
            g = model.own_prof[j]                                       # (C, r)
            R = max(float(_crossing(g[c][None], ro, floor[c])[0]) for c in range(C))
            _paint(out, float(stars.y[i]), float(stars.x[i]), R, g, so)
    idx = np.flatnonzero(model.amp.max(axis=1) > 0)
    if len(idx) == 0 or len(model.shapes) == 0:
        return out
    pos = model.pos[idx]
    b = np.floor(pos).astype(int)
    f = (pos - b)[:, None, None]
    G = (1 - f) * model.shapes[b] + f * model.shapes[np.minimum(b + 1, len(model.shapes) - 1)]   # (k, C, r)
    A = model.amp[idx]
    reach = np.max([_crossing(A[:, c, None] * G[:, c], rr, floor[c]) for c in range(C)], axis=0)
    step = float(rr[1] - rr[0]) if len(rr) > 1 else 1.0
    for j, i in enumerate(idx):
        _paint(out, float(stars.y[i]), float(stars.x[i]), float(reach[j]), G[j] * A[j][:, None], step)
    return out


def _paint(out: np.ndarray, cy: float, cx: float, R: float, g: np.ndarray, step: float) -> None:
    """一顆星的剖面 g (C, r)（每 step 像素一點）畫到 out 上，畫到半徑 R。"""
    if R <= 0:
        return
    h, w = out.shape[1:]
    y0, y1 = int(max(0, np.floor(cy - R))), int(min(h, np.ceil(cy + R) + 1))
    x0, x1 = int(max(0, np.floor(cx - R))), int(min(w, np.ceil(cx + R) + 1))
    if y1 <= y0 or x1 <= x0:
        return
    last = g.shape[1] - 1
    yy, xx = np.ogrid[y0:y1, x0:x1]
    t = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / step
    k0 = np.minimum(t.astype(int), last)
    k1 = np.minimum(k0 + 1, last)
    fr = (t - k0).astype(np.float32)
    out[:, y0:y1, x0:x1] += np.where(t <= last, g[:, k0] * (1 - fr) + g[:, k1] * fr, 0.0)


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


def saturated_tops(hp: np.ndarray, scale: float) -> np.ndarray:
    """飽和的平頂（flat_tops）外擴 1 px：星心落在這裡的星當飽和。"""
    return ndimage.binary_dilation(flat_tops(hp, scale), iterations=1)


def star_like(L: np.ndarray, stars, scale: float, hp: np.ndarray | None = None,
              flat: np.ndarray | None = None) -> np.ndarray:
    """找星結果裡每顆星是不是真的星：各方向對稱（二重不對稱 < ANISOTROPY），而且夠尖銳（≥ SHARPNESS）
    或是飽和的平頂（平頂不尖銳）。回傳 bool 陣列。hp（扣掉背景的 L）、flat（saturated_tops）有算好的話傳進來。"""
    if len(stars.y) == 0:
        return np.zeros(0, bool)
    hp = L - _background(L) if hp is None else hp
    fine = L - ops.blur(L, max(1.5 / max(scale, 1e-6), 1.0))
    h, w = L.shape
    fmax = ndimage.maximum_filter(fine, size=3)
    hmax = ndimage.maximum_filter(hp, size=3)
    y = np.clip(np.round(stars.y).astype(int), 0, h - 1)
    x = np.clip(np.round(stars.x).astype(int), 0, w - 1)
    sharp = fmax[y, x] / np.maximum(hmax[y, x], 1e-12) >= sharp_limit(scale)
    round_ = anisotropy(L, stars, scale) < ANISOTROPY
    # 飽和的平頂星核尖銳度接近 0：用平頂判斷（找星自己的飽和旗標只認得整張最亮的那顆）
    flat = saturated_tops(hp, scale) if flat is None else flat
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
    cmax = _label_max(Ls, lab, n)
    ok = Ls[py, px] >= 0.25 * cmax[comp]
    py, px, comp = py[ok], px[ok], comp[ok]
    has = np.zeros(n + 1, bool)
    has[comp] = True
    h, w = L.shape
    # 3×3 加權重心（小數位置；影像邊上只用裡面的像素）
    oy, ox = np.mgrid[-1:2, -1:2].reshape(2, 1, 9)
    gy, gx = py[:, None] + oy[0], px[:, None] + ox[0]
    inb = (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)
    win = Ls[np.clip(gy, 0, h - 1), np.clip(gx, 0, w - 1)]
    wt = (win - np.where(inb, win, np.inf).min(axis=1, keepdims=True)) * inb
    tot = wt.sum(axis=1)
    tot = np.where(tot > 0, tot, 1.0)
    yy = np.where(wt.sum(axis=1) > 0, (gy * wt).sum(axis=1) / tot, py).astype(np.float64)
    xx = np.where(wt.sum(axis=1) > 0, (gx * wt).sum(axis=1) / tot, px).astype(np.float64)
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
    by, bx = np.nonzero(bright)
    li = lab[by, bx]
    sizes = np.bincount(li, minlength=n + 1)[1:]
    # 補洞只用遮罩邊上的像素（調和內插）：貼著遮罩（2 px 內）的才要遮。以前的補法會用到遮罩外約 9 px，
    # M31 一對離遮罩 5 px 的雙星被抹進補的區域，變成一塊灰白色的盾牌形光斑，那時候範圍是 12 px
    near = ndimage.binary_dilation(mask, iterations=int(round(2 / max(scale, 1e-6))) + 1)
    touching = np.bincount(li, near[by, bx], minlength=n + 1)[1:] > 0
    within = np.bincount(li, ~mask[by, bx], minlength=n + 1)[1:] > 0          # 還沒有整個被遮住
    small = sizes <= 300 / max(scale, 1e-6) ** 2
    cand = np.flatnonzero(touching & within & small)
    keep = np.zeros(n + 1, bool)
    if len(cand):
        # 要像星：尖銳（星雲的亮結、纖維片段平緩，不能納入）
        sh = np.full(n + 1, -np.inf)
        np.maximum.at(sh, li, sharpness(L, scale, hp)[by, bx])
        cand = cand[sh[cand + 1] >= sharp_limit(scale)]
    if len(cand):
        # 而且要圓（星雲纖維上的亮結一邊亮一邊暗，NGC6888 的纖維沿著遮罩一段段被補掉）
        from types import SimpleNamespace
        sel = np.isin(li, cand + 1)
        cy, cx, cl, cv = by[sel], bx[sel], li[sel], hp[by[sel], bx[sel]]
        o = np.lexsort((-cv, cl))                    # 每塊最亮的像素排在最前面
        first = np.concatenate([[True], cl[o][1:] != cl[o][:-1]])
        top = o[first]
        pts = SimpleNamespace(y=cy[top].astype(float), x=cx[top].astype(float))
        keep[cl[top]] = anisotropy(L, pts, scale) < ANISOTROPY
    return ndimage.binary_dilation(keep[lab], iterations=2)


TEXTURE_SCALE = 6.0   # 紋理：幾 px（完整尺寸）以下的起伏從別處搬（AI 降噪後的噪聲一團一團的，2～16 px 都有）
TEXTURE_TILE = 48     # 一次搬一塊幾 px（完整尺寸）


def _masked_blur(planes: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """只用 valid 的像素模糊（正規化卷積），(C, H, W)。"""
    v = valid.astype(np.float32)
    den = np.maximum(ops.blur(v, sigma), 1e-6)
    return np.stack([ops.blur((p * v).astype(np.float32), sigma) / den for p in planes]).astype(np.float32)


AI_FILL_MIN = 6.0     # AI 只補最深處離洞邊 AI_FILL_MIN px（完整尺寸）以上的洞


def big_holes(mask: np.ndarray, scale: float) -> np.ndarray:
    """大的洞（整個連通的洞，只要有一處離洞邊 AI_FILL_MIN px 以上）。"""
    r = max(1, int(round(AI_FILL_MIN / max(scale, 1e-6))))
    core = ndimage.binary_erosion(mask, iterations=r)
    return ndimage.binary_propagation(core, mask=mask) if core.any() else core


def fill_holes(base: np.ndarray, mask: np.ndarray, scale: float, cancel=lambda: None) -> np.ndarray:
    """洞用周圍補（調和內插），但邊界先稍微去噪（洞外 1 px 的模糊，只用洞外的像素）：調和內插直接用邊上的像素，
    邊上一個噪點會變成洞裡一塊平滑的亮斑或暗斑。只模糊 1 px：模糊 6 px 的話星雲結構強的地方邊界跟實際差很多，
    每個洞的邊都看得到（NGC6888）。洞外照原樣。"""
    B = _masked_blur(base, ~mask, 1.0 / max(scale, 1e-6))
    return np.where(mask[None], backend.harmonic_fill(B, mask, cancel), base).astype(np.float32)


def _noise_curve(level: np.ndarray, noise: np.ndarray):
    """一塊塊的（亮度, 噪聲）→ 噪聲隨亮度的曲線：依亮度分 20 組，每組取比較低的 30%（有星雲結構、漏掉的暗星的塊
    量到的「噪聲」偏大），只准隨亮度變大（光子噪聲）。回傳 f(亮度)。"""
    ok = np.isfinite(level) & np.isfinite(noise) & (noise > 0)
    level, noise = level[ok], noise[ok]
    if len(level) < 8:
        m = float(np.median(noise)) if len(noise) else 1e-12
        return lambda v: np.full(np.shape(v), max(m, 1e-12), np.float32)
    order = np.argsort(level)
    groups = np.array_split(order, min(20, max(2, len(order) // 8)))
    lv = np.array([np.median(level[g]) for g in groups])
    nz = np.maximum.accumulate(np.array([np.percentile(noise[g], 30) for g in groups]))
    return lambda v: np.maximum(np.interp(v, lv, nz), 1e-12).astype(np.float32)


def match_texture(base: np.ndarray, mask: np.ndarray, filled: np.ndarray, scale: float, seed: int = 12345) -> np.ndarray:
    """洞裡補上這張圖自己的噪聲紋理，回傳 filled＋紋理（洞外不動）。
    紋理＝洞外 TEXTURE_SCALE px 以下的起伏（高通，只用洞外的像素算），一塊塊（TEXTURE_TILE px）搬進洞裡：每一塊在
    16 個隨機的地方裡挑亮度（噪聲強度）接近、沒有星雲結構、搬過去的像素都不在別的洞裡的，再乘上兩邊噪聲強度的比
    （光子噪聲隨亮度變；噪聲強度用「噪聲隨亮度的曲線」，不直接量：星雲裡直接量會被結構撐大，從暗天空搬來的噪聲
    被放大好幾倍，星雲邊上一片暗的碎點，NGC6888）。洞邊 2 px 內漸入（邊上的噪聲調和內插已經帶進來一些）。
    以前補白噪聲（強度用整張天空量一個，三色同一份）：AI 降噪後真正的噪聲是相關的（相鄰像素相關 +0.3），
    白噪聲是 −0.1，強拉伸後每個洞都是一塊顆粒細、看起來偏暗的圓（LBN576 的亮星）。"""
    C, h, w = base.shape
    if not mask.any():
        return filled
    px = 1.0 / max(scale, 1e-6)
    s = TEXTURE_SCALE * px
    D = base - _masked_blur(base, ~mask, s)
    # 可以搬的地方：洞外 2 px 以外（高通只用洞外的像素算，洞邊也量得對；外擴 6 px 的話密集星場剩不到一半）
    src_ok = ~ndimage.binary_dilation(mask, iterations=max(1, int(round(2 * px))))
    if src_ok.mean() < 0.05:
        return filled
    T = max(4, int(round(TEXTURE_TILE * px)))
    ny, nx = -(-h // T), -(-w // T)
    pad = ((0, ny * T - h), (0, nx * T - w))
    lev = ops.blur(filled.mean(axis=0).astype(np.float32), 2 * s)          # 亮度（洞裡是補過的）

    sub = max(1, T // 16)                    # 每塊的統計只取一部分像素（中位數夠穩，省時間）

    def tiles(a, fill):
        t = np.pad(a.astype(np.float32), pad, constant_values=fill).reshape(ny, T, nx, T)
        return t[:, ::sub, :, ::sub].transpose(0, 2, 1, 3).reshape(ny * nx, -1)
    okt = tiles(src_ok, 0) > 0
    dl = np.where(okt, tiles(D.mean(axis=0), 0), np.nan)
    with warnings.catch_warnings():                 # 整塊都不能用的塊全是 NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mad = 1.4826 * np.nanmedian(np.abs(dl - np.nanmedian(dl, axis=1, keepdims=True)), axis=1)
        levt = np.nanmedian(np.where(okt, tiles(lev, 0), np.nan), axis=1)
    enough = okt.mean(axis=1) > 0.3
    curve = _noise_curve(levt[enough], mad[enough])
    # 來源不要有星點（沒找到的暗星、快速預覽上去不掉的星）：高通後是一個亮點＋一圈負的，搬進洞裡就是複製一顆星
    # 和幾個黑點（LBN576 快速預覽，亮星的洞裡）。超過 4 倍當地噪聲的像素外擴 2 px 都不搬
    noise_here = curve(lev)
    spot = np.abs(D.mean(axis=0)) > 4.0 * noise_here
    src_ok &= ~ndimage.binary_dilation(spot, iterations=max(1, int(round(2 * px))) + 1)
    levt_all = tiles(lev, 0).mean(axis=1)
    fl = np.log(curve(levt_all))
    struct = np.where(enough, np.abs(np.log(np.maximum(mad, 1e-30)) - np.log(curve(levt))), np.inf)
    rng = np.random.default_rng(seed)
    K = 16
    cy = rng.integers(0, ny, (ny * nx, K))
    cx = rng.integers(0, nx, (ny * nx, K))
    ys, xs = np.nonzero(mask)
    tyi, txi, oy, ox = ys // T, xs // T, ys % T, xs % T
    tid = tyi * nx + txi
    # 每個候選：這一塊洞裡的像素搬過去的地方有多少可以用（剛好是別的星的洞就不行）。一塊裡的像素盡量從同一個地方搬，
    # 東拼西湊的話相鄰像素來自不同地方、不相關，紋理比周圍碎
    def usable(i, k):
        sy, sx = cy[tid[i], k] * T + oy[i], cx[tid[i], k] * T + ox[i]
        return (sy < h) & (sx < w) & src_ok[np.minimum(sy, h - 1), np.minimum(sx, w - 1)]
    probe = np.arange(0, len(ys), 5)                 # 評分只用一部分像素
    count = np.maximum(np.bincount(tid[probe], minlength=ny * nx), 1)
    frac = np.stack([np.bincount(tid[probe], weights=usable(probe, k), minlength=ny * nx) for k in range(K)],
                    axis=1) / count[:, None]
    src = cy * nx + cx
    score = np.abs(fl[src] - fl[:, None]) + struct[src] + 3.0 * (1.0 - frac)
    score[src == np.arange(ny * nx)[:, None]] = np.inf
    order = np.argsort(score, axis=1)
    nt = curve(lev[ys, xs])
    tex = np.zeros((C, len(ys)), np.float32)
    done = np.zeros(len(ys), bool)
    for r in range(K):                      # 先用最好的候選，搬不到的像素換下一個
        left = np.flatnonzero(~done)
        if len(left) == 0:
            break
        k = order[tid[left], r]
        g = left[usable(left, k)]
        k = order[tid[g], r]
        sy = cy[tid[g], k] * T + oy[g]
        sx = cx[tid[g], k] * T + ox[g]
        ratio = np.clip(nt[g] / curve(lev[sy, sx]), 0.25, 4.0)
        for c in range(C):                  # 保險：再限制在 ±8 倍噪聲（各色版的噪聲比亮度大，不切到真的噪聲）
            tex[c, g] = np.clip(D[c, sy, sx] * ratio, -8.0 * nt[g], 8.0 * nt[g])
        done[g] = True
    rest = np.flatnonzero(~done)            # 都找不到：白噪聲
    if len(rest):
        z = rng.standard_normal(len(rest)).astype(np.float32)
        for c in range(C):
            tex[c, rest] = z * nt[rest]
    # 洞邊漸入：最外一圈 1/3、第二圈 2/3（完整尺寸；快速預覽的一個像素已經是 4 px，不用漸入）
    if px > 0.5:
        inner1 = ndimage.binary_erosion(mask, iterations=1)
        inner2 = ndimage.binary_erosion(inner1, iterations=1)
        ramp = ((1.0 + inner1[ys, xs] + inner2[ys, xs]) / 3.0).astype(np.float32)
    else:
        ramp = 1.0
    out = filled.copy()
    out[:, ys, xs] += tex * ramp
    return out


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
        return {"threshold": 5.0, "grow": 2.0, "texture": True, "spikes": True, "fringes": True, "ai_fill": True}

    def controls(self) -> list[dict]:
        return [
            slider("threshold", T("偵測門檻", "Detection threshold"), 2, 12, 0.5, "{:.1f} σ",
                   help=T("數字小會抓到更暗的星，但也可能把星雲的亮點當成星。", "Lower finds fainter stars but may catch "
                                                                    "nebula knots.")),
            slider("grow", T("範圍外擴", "Grow"), 0, 12, 0.5, "{:.1f} px",
                   help=T("無星圖上留下星暈的話調大。", "Increase if halos remain on the starless image.")),
            toggle("ai_fill", T("用 AI 補星點的洞", "Fill star holes with AI"),
                   help=T("用周圍的樣子補（延續星雲的紋路、亮暗）；關掉的話用平滑的內插。沒有 AI 模型時自動用內插。",
                          "Continues the surrounding nebula structure into the holes; off uses smooth interpolation. "
                          "Falls back to interpolation when the AI model is missing.")),
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
        stars = detect_stars(L, float(params.get("threshold", 5.0)))
        stars = recenter_stars(L, stars, ctx.scale)
        hp = L - _background(L)
        flat = saturated_tops(hp, ctx.scale)
        # 星雲的亮結也會被找星抓到：不夠尖銳的不當星（留在無星圖）
        like = star_like(L, stars, ctx.scale, hp, flat)
        ctx.view["not_stars"] = int((~like).sum())
        stars = keep_stars(stars, like)
        ctx.check()
        # 星點模型：星核挖掉補，外圍扣模型（見 star_model）。後面量星暈、繞射芒、暗環都在扣過模型的影像上量
        grow = float(params.get("grow", 2.0)) / ctx.scale
        model = star_model(img, stars, ctx.scale, hp, flat)
        floor = 0.05 * np.array([ops.pixel_noise(p) for p in img])
        wings = render_stars(model, stars, L.shape, floor)
        img_all, L_all = img, L
        img = img - wings
        L = img.mean(axis=0)
        core_r = model.hole
        mask = _discs(L.shape, stars.y, stars.x, core_r + grow)
        ctx.check()
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
        ctx.view["holes"] = mask                       # 實際補掉的範圍（給檢查工具、之後介面顯示用）
        ctx.progress(0.4, "inpaint")
        if halos is not None:
            # 星暈淡的部分少扣一點（r²／(r＋0.3σ)），無星圖扣多少、星點層就放多少：星暈外圍量到噪聲裡
            # （AI 降噪後量得很遠），整個扣的話中等亮度的星周圍各有一圈淡淡的光盤；只在星點層打折的話，
            # 淡的星暈兩邊都沒了，合成後每顆星的位置一個暗洞。以前是 r²／(r＋σ)：星暈再亮都留 1σ，亮星整個洞偏亮
            for c in range(img.shape[0]):
                n = FOOT_LEVEL * ops.pixel_noise(img[c])
                halos[c] = halos[c] * halos[c] / (halos[c] + n + 1e-12)
        base = img - halos if halos is not None else img
        smooth = fill_holes(base, mask, ctx.scale, ctx.check)
        ctx.check()
        if params.get("ai_fill", True):
            from . import ai as tai
            big = big_holes(mask, ctx.scale)
            if tai.available_inpaint() and big.any():
                # 平滑內插在大的洞（亮星、星暈）中間是一塊沒有紋路的平面；AI 延續周圍星雲的樣子。
                # 小的洞內插＋噪聲紋理就跟周圍一樣了（第二、三步之後量的），不用 AI
                notes = state.notes.get("main", {})
                ratio = notes.get("pixel_noise_ratio") or (notes.get("denoise_ratio") or (1.0,))[0]
                smooth = tai.inpaint_planes(base, mask, smooth, ctx, float(ratio), only=big)
                ctx.check()
        # 星點層＝原圖 − 補好的背景，但先扣掉 1 倍噪聲再去負值：直接去負值的話正的噪聲留下、負的切掉，
        # 每顆星周圍多一個帶噪聲的亮盤（遮罩範圍），合成後變成一圈光暈。遮罩邊緣漸變
        # 遮罩邊緣往內 4 px 漸變、噪聲門檻用平滑的曲線（r²／(r＋σ)，大的值≈r−σ），不然星點層拉伸後
        # 光暈在遮罩邊緣突然切掉，星點外圍看起來有一圈邊
        resid = img_all - smooth
        # 遮罩邊緣往內的距離（只需要到 4 px；整張的距離轉換很慢）
        reach = int(np.ceil(4.0 / ctx.scale))
        dist = np.zeros(mask.shape, np.float32)
        inner = mask
        for _k in range(reach + 1):
            dist += inner
            inner = ndimage.binary_erosion(inner, iterations=1)
            if not inner.any():
                break
        edge = ops.smoothstep(0.0, 4.0 / ctx.scale, dist).astype(np.float32)
        layers = []
        for c in range(img.shape[0]):
            # 淡的部分（幾倍噪聲以內）先模糊再取：星點外圍那圈光只有幾個 σ，噪聲一抖一半的像素變 0，星點層拉伸很強，
            # 外圍變成亮暗交錯的一圈碎點、合成後像黑圈（LBN576 亮星，2026-10-19）。亮的星核照原本的值，不會變糊
            n = ops.pixel_noise(img[c])
            rb = ops.blur(resid[c].astype(np.float32), 1.2 / ctx.scale)
            sharp = ops.smoothstep(3.0 * n, 12.0 * n, rb)
            r = np.maximum(resid[c] * sharp + rb * (1.0 - sharp), 0)
            soft = r * r / (r + n + 1e-12) * edge + wings[c] * (1.0 - edge)
            if halos is not None:
                soft += halos[c] * (1.0 - edge)
            layers.append(soft)
        stars_layer = np.stack(layers).astype(np.float32)
        if img.shape[0] >= 3 and params.get("fringes", True):
            stars_layer = smooth_star_colour(stars_layer, 2.0 / ctx.scale)
        starless = smooth
        if params.get("texture", True):
            starless = match_texture(base, mask, smooth, ctx.scale)
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
