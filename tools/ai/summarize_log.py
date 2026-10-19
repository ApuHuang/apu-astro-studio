"""把 train_n2n 的 log 整理成一行一行的驗證結果（原圖 → AI）。python tools/ai/summarize_log.py <log>"""
import ast
import sys

for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if line.startswith("val"):
        d = ast.literal_eval(line.split(":", 1)[1].strip())
        out = []
        for k, v in d.items():
            parts = [f"{m} {v['raw_' + m]}→{v['ai_' + m]}" for m in ("sky", "faint", "bright", "core") if "raw_" + m in v]
            out.append(f"{k}: " + ", ".join(parts))
        print(" | ".join(out))
    elif line.startswith(("best", "export")):
        print("  " + line)
