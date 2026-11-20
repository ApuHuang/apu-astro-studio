"""契約：Studio 用到的每一個 APU Processing 函式（apu_processing.api），用跟 Studio 一樣的呼叫方式。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from apu_processing import api as papi

from .data import star_field
from .snap import Display, check


@pytest.fixture(scope="module")
def img() -> np.ndarray:
    return star_field()


def lum(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=0)


def test_api_names():
    check("processing_api_names", sorted(papi.__all__))


# ---------------------------------------------------------------------- 讀寫檔


def test_fits_roundtrip(tmp_path, img):
    p = tmp_path / "m.fits"
    hdr = {"FILTER": "L", "OBJECT": "Test"}
    from astropy.io import fits
    papi.save_fits(p, img, fits.Header(hdr))
    data, header = papi.load_image(p)
    check("processing_fits_roundtrip", {"data": data, "filter": header.get("FILTER"), "suffixes": sorted(papi.FITS_SUFFIXES)})


def test_load_tiff16(tmp_path, img):
    import tifffile
    p = tmp_path / "m.tif"
    tifffile.imwrite(p, np.round(np.moveaxis(img, 0, -1) * 65535).astype(np.uint16), photometric="rgb")
    data, _header = papi.load_image(p)
    check("processing_load_tiff16", data)


# ---------------------------------------------------------------------- 疊圖紀錄、覆蓋率、合成


def test_coverage(tmp_path, img):
    from astropy.io import fits
    master = tmp_path / "master.fits"
    papi.save_fits(master, img)
    frac = np.ones(img.shape[-2:], np.float32) * 10
    frac[:, :12] = 3
    frac[-8:, :] = 0
    fits.PrimaryHDU(frac).writeto(tmp_path / "master_coverage.fits")
    found = papi.find_coverage(master)
    f = papi.load_coverage_fraction(found)
    cov = papi.combine_coverage([f, f * 0.95], img.shape[-2:])
    good = papi.coverage_good_mask(cov)
    crop = papi.suggest_crop(good.astype(np.float32))
    check("processing_coverage", {"found": found, "missing": papi.find_coverage(tmp_path / "other.fits"),
                                  "fraction": f, "combined": cov, "good": good, "crop": crop,
                                  "one_missing": papi.combine_coverage([f, None], img.shape[-2:]),
                                  "wrong_shape": papi.combine_coverage([f[:-1]], img.shape[-2:])}, tmp_path)


def test_recipe(tmp_path, img):
    for name in ("a_Ha.fits", "a_OIII.fits", "b_L.fits"):
        papi.save_fits(tmp_path / name, img[0])
    rec = tmp_path / "t.recipe.json"
    rec.write_text(json.dumps({"target": "T", "outputs": [
        {"file": "a_Ha.fits", "filter": "Ha", "align_group": "A", "frames": 10},
        {"file": "a_OIII.fits", "filter": "OIII", "align_group": "A", "frames": 12},
        {"file": "b_L.fits", "filter": "L", "align_group": "B", "frames": 5},
        {"file": "missing.fits", "filter": "S", "align_group": "A"}]}), encoding="utf-8")
    project = papi.load_recipe(rec)
    group, outputs = project.main_group()
    check("processing_recipe", {"is": [papi.is_recipe(rec), papi.is_recipe(tmp_path / "a_Ha.fits")],
                                "target": project.target, "group": group,
                                "outputs": [(o.path, o.filter, o.group, o.frames, o.coverage) for o in outputs]},
          tmp_path)


def test_compose(img):
    planes = [img[0] * 1.0, img[1] * 0.6 + 0.01, img[2] * 0.4 + 0.02]
    filters = ["Ha", "OIII", "SII"]
    composer = papi.Composer(planes, ["ha.fits", "oiii.fits", "sii.fits"])
    rec = papi.ComposeSettings.recommended(filters)
    hoo = papi.ComposeSettings.for_preset("HOO", filters).with_source(0, 0).with_strength(1, 1.5)
    check("processing_compose", {"presets": {k: [list(v) for v in dict(p).values()] if isinstance(p, dict) else p
                                             for k, p in papi.COMPOSE_PRESETS.items()},
                                 "recommended": [rec.preset, [[c.role, c.source, c.strength] for c in rec.channels]],
                                 "hoo": [hoo.preset, [[c.role, c.source, c.strength] for c in hoo.channels]],
                                 "rgb_recommended": composer.compose(rec), "rgb_hoo": composer.compose(hoo)})


# ---------------------------------------------------------------------- 找星、噪聲


def test_find_stars(img):
    L = lum(img)
    out = {}
    for thr in (None, 3.0, 4.0, 5.0, 8.0):
        s = papi.find_stars(L) if thr is None else papi.find_stars(L, thr)
        out[str(thr)] = {"y": s.y, "x": s.x, "radius": s.radius, "saturated": s.saturated, "mask": s.mask,
                         "weight": s.weight, "noise": s.noise}
    check("processing_find_stars", out)


def test_fine_noise_and_fwhm(img):
    L = lum(img)
    stars = papi.find_stars(L)
    check("processing_noise_fwhm", {"fine_noise": papi.fine_noise(L), "fine_noise_mono": papi.fine_noise(img[0]),
                                    "core_fwhm": papi.measure_core_fwhm(L, stars),
                                    "psf_core_limit": papi.PSF_CORE_LIMIT})


def test_stars_constructor():
    """Studio 自己的找星直接建 Stars（6 個位置參數，noise 用預設）。"""
    z = np.zeros(0)
    s = papi.Stars(z, z, z, np.zeros(0, bool), np.zeros((4, 5), bool), np.zeros((4, 5), np.float32))
    check("processing_stars_constructor", {"fields": [f for f in s.__dataclass_fields__], "noise": s.noise})


def test_starlet(img):
    coeffs, smooth = papi.starlet(lum(img), 3)
    check("processing_starlet", {"coeffs": list(coeffs), "smooth": smooth})


# ---------------------------------------------------------------------- 降噪


def test_reduce_noise_legacy(img):
    """Studio 的降噪：Processing 預設值 + 0.2.1 的做法（noise.LEGACY）+ 強度、亮度／色彩倍率。"""
    from apu_astro.studio.tools.noise import LEGACY
    base = papi.DenoiseSettings()
    check("processing_denoise_defaults", base)
    keep = {k: v for k, v in LEGACY.items() if hasattr(base, k)}
    out = {}
    for amount, lu, ch in ((0.5, 1.0, 1.0), (0.3, 0.8, 1.2)):
        s = replace(base, amount=amount, **keep, k_luminance=tuple(k * lu for k in base.k_luminance),
                    k_chroma=tuple(k * ch for k in base.k_chroma))
        out[f"{amount}"] = papi.reduce_noise(img, s, lambda: False)
    out["mono"] = papi.reduce_noise(img[0], replace(base, amount=0.5, **keep), lambda: False)
    check("processing_reduce_noise", out)


# ---------------------------------------------------------------------- 拉伸、校色


def test_measure_stretch_noise(img):
    big = star_field(300, 420, seed=3)
    out = {}
    for scale in (1.0, 2.0, 3.0):
        n = papi.measure_stretch_noise(big, scale)
        out[str(scale)] = n
    out["mono"] = papi.measure_stretch_noise(big[0], 1.0)
    check("processing_measure_stretch_noise", out)


def test_apply_stretch(img):
    big = star_field(300, 420, seed=3)
    noise = papi.measure_stretch_noise(big, 1.0)
    out = {}
    for bg, strength, linked, bgc in ((0.13, 0.5, True, 0.5), (0.2, 0.8, False, 0.0)):
        s = papi.StretchSettings(background=bg, strength=strength, linked=linked, background_color=bgc)
        v = papi.apply_stretch(big, s, noise)
        # 8-bit 差 ≤ 1 抓不到小改變（拉伸常數改 1% 只差 1 階）：另外存一份浮點值，用相對誤差 1e-5 比
        out[f"{bg}_{strength}_{linked}"] = {"display": Display(v), "float": np.asarray(v, np.float32)}
    v = papi.apply_stretch(big[0], papi.StretchSettings(), papi.measure_stretch_noise(big[0]))
    out["mono"] = {"display": Display(v), "float": np.asarray(v, np.float32)}
    check("processing_apply_stretch", out)


def test_measure_balance(img):
    big = star_field(300, 420, seed=3)
    big[0] *= 1.3
    check("processing_measure_balance", papi.measure_balance(big))


# ---------------------------------------------------------------------- 曲線


def test_curves(img):
    disp = np.clip(img * 4, 0, 1)
    curves = {"master": [[0.0, 0.0], [0.4, 0.55], [1.0, 1.0]], "red": [list(p) for p in papi.IDENTITY_CURVE],
              "green": [[0.0, 0.05], [1.0, 0.95]], "blue": [list(p) for p in papi.IDENTITY_CURVE]}
    adj = papi.DisplayAdjustments(curves=papi.ToneCurves(**curves))
    xs = np.linspace(0, 1, 33)
    check("processing_curves", {"identity": [list(p) for p in papi.IDENTITY_CURVE],
                                "values": papi.curve_values(curves["master"], xs),
                                "rgb": Display(papi.apply_display(disp, adj)),
                                "mono": Display(papi.apply_display(disp[0], adj)),
                                "rgb_float": np.asarray(papi.apply_display(disp, adj), np.float32),
                                "mono_float": np.asarray(papi.apply_display(disp[0], adj), np.float32),
                                "default_curves": papi.ToneCurves()})


# ---------------------------------------------------------------------- 其他


def test_i18n_and_cancelled():
    i18n = papi.i18n
    old = i18n.get_language()
    out = {}
    try:
        for lang in ("zh", "en"):
            i18n.set_language(lang)
            out[lang] = [i18n.get_language()] + [i18n.tr(f"gui.curve.{k}") for k in ("master", "red", "green", "blue")]
    finally:
        i18n.set_language(old)
    out["cancelled_is_exception"] = issubclass(papi.Cancelled, Exception)
    check("processing_i18n", out)
