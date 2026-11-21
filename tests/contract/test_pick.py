"""契約：挑片頁（pages/pick*.py）用到的 APU Pick 函式與常數，用跟挑片頁一樣的呼叫方式。"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from astro_light_selector import settings as pick_settings
from astro_light_selector.field import CORNERS, GRID, analyze_frame, combine
from astro_light_selector.grouping import EXPOSURE_TOLERANCE
from astro_light_selector.i18n import set_language
from astro_light_selector.i18n import tr as ptr
from astro_light_selector.i18n import tr_metric
from astro_light_selector.metrics import FrameMetrics, is_image
from astro_light_selector.mover import read_move_log
from astro_light_selector.organize import (DEFAULT_ORGANIZE_KEYS, ORGANIZE_KEYS, organize, pending_undo, plan_organize,
                                           undo_organize)
from astro_light_selector.pipeline import (REJECT_DIR_NAME, REPORT_NAME, Cancelled, Selection, collect_files, decide,
                                           frame_key, measure_files, move_files, restore_files, should_recurse)
from astro_light_selector.preview import Preview, closeup, make_preview, thumbnail
from astro_light_selector.report import (match_frames, read_csv, read_overrides, report_subfolders,
                                         score_summary_lines, write_csv)
from astro_light_selector.scoring import DEFAULT_WEIGHTS, METRICS, ScoreConfig
from astro_light_selector.selector import Decision

from .data import write_frames
from .snap import Display, check


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("pick")
    folder = root / "lights"
    lights, _darks = write_frames(folder, n_light=8, n_dark=0)
    files, home_of = collect_files(folder, folder / REJECT_DIR_NAME, False)
    frames = measure_files(files, 1, home_of=home_of)
    return root, folder, files, home_of, frames


def _selection(frames, folder, method="pass", value=80.0) -> tuple[Selection, ScoreConfig]:
    """挑片頁 _config／_apply 的做法。"""
    cfg = ScoreConfig(pass_pct=80 / 100)
    if method == "min_score":
        cfg.min_score = value
    elif method == "keep_best":
        cfg.keep_best = value / 100
    weights = dict(DEFAULT_WEIGHTS)
    total = sum(weights.values())
    cfg.weights = {m: w / total for m, w in weights.items()}
    sel = decide(frames, ("filter", "exposure"), "score", cfg, exposure_tolerance=EXPOSURE_TOLERANCE,
                 overrides={}, root=folder)
    return sel, cfg


def test_constants():
    set_language("en")
    check("pick_constants", {"report": REPORT_NAME, "reject": REJECT_DIR_NAME, "metrics": list(METRICS),
                             "weights": DEFAULT_WEIGHTS, "tolerance": EXPOSURE_TOLERANCE,
                             "organize_keys": list(ORGANIZE_KEYS), "default_organize": list(DEFAULT_ORGANIZE_KEYS),
                             "grid": GRID, "corners": CORNERS, "metric_names": [tr_metric(m) for m in METRICS],
                             "cancelled_is_exception": issubclass(Cancelled, Exception),
                             "frame_metrics_fields": list(FrameMetrics.__dataclass_fields__),
                             "decision_fields": list(Decision.__dataclass_fields__),
                             "preview_fields": list(Preview.__dataclass_fields__),
                             "score_config": ScoreConfig()})


def test_collect_and_measure(measured):
    root, folder, files, home_of, frames = measured
    check("pick_measure", {"files": [p.name for p in files], "home_of": home_of,
                           "is_image": [is_image(files[0]), is_image(folder / "x.txt")],
                           "should_recurse": should_recurse(folder), "frame_key": frame_key(files[0], folder),
                           "frames": frames}, root)


def test_decide(measured):
    root, folder, files, home_of, frames = measured
    out = {}
    for method, value in (("pass", 80.0), ("min_score", 60.0), ("keep_best", 50.0)):
        sel, _cfg = _selection(frames, folder, method, value)
        out[method] = {"decisions": sel.decisions, "group_sizes": sel.group_sizes, "thresholds": sel.thresholds,
                       "n_keep": sel.n_keep,
                       "summary": [str(x) for r in sel.results.values() if r for x in score_summary_lines(r)]}
    check("pick_decide", out, root)


def test_report_roundtrip(measured, tmp_path):
    root, folder, files, home_of, frames = measured
    work = tmp_path / "lights"
    shutil.copytree(folder, work)
    wfiles, whome = collect_files(work, work / REJECT_DIR_NAME, False)
    wframes = measure_files(wfiles, 1, home_of=whome)
    sel, _cfg = _selection(wframes, work)
    write_csv(sel.decisions, work / REPORT_NAME, root=work)
    back = read_csv(work / REPORT_NAME, work)
    kept, renamed = match_frames(back, wfiles, whome, work)
    check("pick_report", {"read": back, "kept": [Path(f.file).name for f in kept], "renamed": renamed,
                          "overrides": read_overrides(work / REPORT_NAME),
                          "subfolders": report_subfolders(work / REPORT_NAME, work)}, tmp_path)


def test_move_and_restore(measured, tmp_path):
    root, folder, files, home_of, frames = measured
    work = tmp_path / "lights"
    shutil.copytree(folder, work)
    wfiles, whome = collect_files(work, work / REJECT_DIR_NAME, False)
    sel, _cfg = _selection(measure_files(wfiles, 1, home_of=whome), work, "keep_best", 50.0)
    reject = work / REJECT_DIR_NAME
    dry = move_files(sel.decisions, reject, False, dry_run=True)
    done = move_files(sel.decisions, reject, False)
    log = read_move_log(reject)
    after, after_home = collect_files(work, reject, False)
    restored = restore_files(work, reject, False)
    check("pick_move", {"dry": [[p.name for p in x] for x in dry], "done": [[p.name for p in x] for x in done],
                        "log": sorted((a.name, b.parent.name) for a, b in log.items()),
                        "collect_after": sorted(p.name for p in after), "home_after": len(after_home),
                        "restored": sorted(p.name for p in restored)}, tmp_path)


def test_organize(measured, tmp_path):
    root, folder, files, home_of, frames = measured
    work = tmp_path / "lights"
    shutil.copytree(folder, work)
    wfiles, whome = collect_files(work, None, True)
    wframes = measure_files(wfiles, 1, home_of=whome)
    plan = plan_organize(wframes, work, tuple(DEFAULT_ORGANIZE_KEYS), EXPOSURE_TOLERANCE)
    moved = organize(plan, work)
    pend = pending_undo(work)
    undone = undo_organize(work)
    check("pick_organize", {"plan": [(a.relative_to(work).as_posix(), b.relative_to(work).as_posix())
                                     for a, b in plan],
                            "moved": len(moved), "pending": pend, "undone": len(undone),
                            "after_undo": sorted(p.name for p in work.glob("*.fits"))})


def test_preview(measured):
    root, folder, files, home_of, frames = measured
    p = make_preview(files[0])
    thumb = thumbnail(p, 120, 90)
    cu = closeup(p, 0.5, 0.5, 40, 2)
    check("pick_preview", {"preview": {**{f: getattr(p, f) for f in Preview.__dataclass_fields__ if f != "image"},
                                       "image": Display(p.image),
                                       "image_float": np.asarray(p.image, np.float32)},
                           "thumb": Display(np.asarray(thumb, np.float32) / 255),
                           "closeup": Display(np.asarray(cu, np.float32) / 255)})


def test_field(measured):
    root, folder, files, home_of, frames = measured
    results = [analyze_frame(p) for p in files[:3]]
    avg = combine(results)
    check("pick_field", {"one": results[0], "avg": avg}, root)


def test_detail_lines(measured, app):
    from apu_astro.pages.pick_detail import pick_detail
    root, folder, files, home_of, frames = measured
    set_language("en")
    detail = pick_detail()
    sel, cfg = _selection(frames, folder)
    d = sel.decisions[0]
    preview = make_preview(files[0])
    info = detail.info_lines(d, sel.results.get(d.group), cfg.weights,
                             [x for x in sel.decisions if x.group == d.group], preview, "", sel.thresholds.get(d.group))
    field = combine([analyze_frame(p) for p in files[:3]])
    check("pick_detail_lines", {"info": info, "histogram": detail.histogram_stats(preview),
                                "field_grid": detail.field_lines(field, "grid"),
                                "field_quad": detail.field_lines(field, "quad"),
                                "field_none": detail.field_lines(None, "grid")}, root)


def test_settings_and_tr():
    pick_settings.save_settings(preview_pane="info", group_keys=["filter"])
    pick_settings.save_settings(preview_pane="hist")
    s = pick_settings.load_settings()
    out = {"merged": {k: s.get(k) for k in ("preview_pane", "group_keys")}}
    for lang in ("zh", "en"):
        set_language(lang)
        out[lang] = [ptr("gui.status.found", n=3), ptr("gui.preview.closeup"), ptr("summary.no_measurable")]
    set_language("en")
    check("pick_settings_tr", out)
