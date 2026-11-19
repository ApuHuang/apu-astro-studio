"""Studio 只透過 apu_processing.api 引用 APU Processing（其他模組是 Processing 的內部實作，可能隨時改）。"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "apu_astro"
ALLOWED = re.compile(r"from apu_processing import api as \w+$|from apu_processing\.api import ")


def test_only_api_imports():
    bad = []
    for path in SRC.rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if re.match(r"(from|import) apu_processing\b", s) and not ALLOWED.match(s):
                bad.append(f"{path.relative_to(SRC)}:{n}: {s}")
    assert not bad, "\n".join(bad)
