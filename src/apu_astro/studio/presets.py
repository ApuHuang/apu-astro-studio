"""預設集與步驟清單的分組（引擎這邊，不 import Qt）。

- 預設集：「標準」（新的預設）、「阿噗流程」（進階）、使用者存的自訂預設集。
  標準與阿噗流程用的是同一串步驟（chain.default_steps，成品一樣），差在介面：
  標準把步驟摺疊成 4 個階段、每個階段只露出關鍵滑桿（可以展開看細步驟）；阿噗流程列出全部步驟、去星後的分支分組
- 分支：去星之後到合成星點之前，每一步作用在無星圖、星點層或兩者（branch_of）；清單依分支分組顯示。
  不同分支的步驟互不影響，所以分組顯示不改變計算順序的意義
- 階段：線性處理／去星與拉伸／調色與細節／合成（stage_of），依去星、拉伸、合成星點的位置分
"""

from __future__ import annotations

import uuid

from . import chain as ch
from .tools import T, get

STANDARD = "standard"
APU = "apu"
CUSTOM = "custom:"
BUILTIN = {STANDARD: T("標準", "Standard"), APU: T("阿噗流程", "APU workflow")}
VIEW_SIMPLE, VIEW_FULL = "simple", "full"

STAGES = [T("線性處理", "Linear processing"), T("去星與拉伸", "Star removal & stretch"),
          T("調色與細節", "Colour & detail"), T("合成", "Recombine")]

# 每種工具在「標準」裡露出的關鍵滑桿（其他參數展開到細步驟才看得到）
KEY_PARAMS: dict[str, list[str]] = {
    "gradient": ["smooth"],
    "color_cal": ["strength"],
    "deconv": ["iterations"],
    "ai_denoise": ["amount"],
    "star_split": ["grow"],
    "denoise": ["amount"],
    "stretch": ["strength", "background"],
    "color_grade": ["vibrance", "saturation"],
    "sharpen": ["detail", "structure"],
    "star_merge": ["brightness", "reduce", "saturation"],
}
STRETCH_KEYS = {"auto": ["strength", "background"], "arcsinh": ["beta"], "ghs": ["ghs_d", "ghs_b"],
                "stf": ["stf_target"]}


def key_params(step: ch.Step) -> list[str]:
    if step.kind == "stretch":
        return STRETCH_KEYS.get(step.params.get("method", "auto"), [])
    return KEY_PARAMS.get(step.kind, [])


def view_of(preset: str | None) -> str:
    return VIEW_SIMPLE if preset == STANDARD else VIEW_FULL


def title_of(preset: str | None) -> str:
    if preset in BUILTIN:
        return str(BUILTIN[preset])
    if preset and preset.startswith(CUSTOM):
        return preset[len(CUSTOM):]
    return str(T("自訂", "Custom"))


def template(preset: str, color: bool, narrowband: str | None, custom: dict | None = None) -> list[ch.Step]:
    """預設集的步驟（新的 id）。自訂預設集找不到時回到標準。"""
    if preset.startswith(CUSTOM):
        data = (custom or {}).get(preset[len(CUSTOM):])
        if data:
            return steps_from_dicts(data.get("steps", []))
    return ch.default_steps(color, narrowband)


def steps_to_dicts(steps: list[ch.Step]) -> list[dict]:
    """存成預設集用：不含 id（套用時給新的）。"""
    out = []
    for s in steps:
        d = s.to_dict()
        d.pop("id", None)
        out.append(d)
    return out


def steps_from_dicts(items: list[dict]) -> list[ch.Step]:
    out = []
    for d in items:
        if d.get("kind") not in _kinds():
            continue
        out.append(ch.Step.from_dict({**d, "id": uuid.uuid4().hex[:8]}))
    return out


def same_steps(a: list[ch.Step], b: list[ch.Step]) -> bool:
    """兩串步驟的內容一樣（不管 id）。"""
    return steps_to_dicts(a) == steps_to_dicts(b)


def _kinds() -> set[str]:
    from .tools import all_tools
    return {t.kind for t in all_tools()}


# ---------------------------------------------------------------------- 分支與階段


def branch_of(steps: list[ch.Step]) -> list[str | None]:
    """每一步屬於哪個分支：None（去星前、合成後，或沒有去星）、"split"（去星本身）、"starless"／"stars"／"both"、
    "merge"（合成星點，匯合點）。停用的去星／合成星點不算分支點。"""
    out: list[str | None] = []
    sep = False
    for s in steps:
        if s.kind == "star_split" and s.enabled:
            out.append("split")
            sep = True
        elif s.kind == "star_merge" and s.enabled and sep:
            out.append("merge")
            sep = False
        elif not sep:
            out.append(None)
        elif s.layer in ("starless", "stars"):
            out.append(s.layer)
        elif s.layer == "main":
            out.append("both")
        else:
            auto = get(s.kind).auto_layer
            out.append("both" if auto == "both" else auto if auto in ("starless", "stars") else "starless")
    return out


def display_order(steps: list[ch.Step]) -> list[tuple[str, object]]:
    """完整清單要顯示的列：("step", 第幾步, 分支或 None)、("header", 分支)。
    去星後到合成星點前的步驟依分支分組（無星圖、星點、兩者），組內照原本的順序。"""
    br = branch_of(steps)
    rows: list[tuple[str, object]] = []
    i, n = 0, len(steps)
    while i < n:
        rows.append(("step", i, None if br[i] in (None, "split", "merge") else br[i]))
        if br[i] == "split":
            j = i + 1
            while j < n and br[j] not in ("merge", None):
                j += 1
            span = list(range(i + 1, j))
            for name in ("starless", "stars", "both"):
                members = [k for k in span if br[k] == name]
                if members:
                    rows.append(("header", name))
                    rows += [("step", k, name) for k in members]
            i = j
            continue
        i += 1
    return rows


def display_numbers(steps: list[ch.Step]) -> list[int]:
    """每一步在畫面上的編號（1 起）：照完整清單的顯示順序（去星後依分支分組）重新編號，所以清單上是連續的。
    只是顯示用：步驟鏈的實際順序、專案檔存的內容都不變。"""
    out = [0] * len(steps)
    n = 0
    for row in display_order(steps):
        if row[0] == "step":
            n += 1
            out[int(row[1])] = n
    return out


def neighbour_in_branch(steps: list[ch.Step], index: int, d: int) -> int | None:
    """上移／下移：同一組（同分支、同一段）裡的上一個／下一個步驟；分組外就是清單上的鄰居。"""
    order = [r for r in display_order(steps) if r[0] == "step"]
    pos = next((k for k, r in enumerate(order) if r[1] == index), None)
    if pos is None:
        return None
    k = pos + d
    if not 0 <= k < len(order):
        return None
    if order[k][2] != order[pos][2]:
        return None
    return int(order[k][1])


def stage_of(steps: list[ch.Step]) -> list[int]:
    """每一步屬於哪個階段（0 線性處理、1 去星與拉伸、2 調色與細節、3 合成）。"""
    n = len(steps)
    kinds = [s.kind for s in steps]
    split = kinds.index("star_split") if "star_split" in kinds else None
    stretches = [i for i, k in enumerate(kinds) if k == "stretch"]
    merge = max((i for i, k in enumerate(kinds) if k == "star_merge"), default=None)
    start2 = split if split is not None else (stretches[0] if stretches else n)
    end2 = max([start2] + [i for i in stretches if i >= start2])
    out = []
    for i in range(n):
        if merge is not None and i >= merge:
            out.append(3)
        elif i < start2:
            out.append(0)
        elif i <= end2:
            out.append(1)
        else:
            out.append(2)
    return out


def stage_steps(steps: list[ch.Step], stage: int) -> list[int]:
    return [i for i, s in enumerate(stage_of(steps)) if s == stage]
