import re
import html
from collections import defaultdict

LOG_FILE = "all_bad_op_occurrences.log"
BAD_OPS_FILE = "bad_ops.txt"
OUT_FILE = "failed_op_report.html"

with open(BAD_OPS_FILE, "r", encoding="utf-8") as f:
    bad_ops = {line.strip() for line in f if line.strip()}

# 每个 bad op 收集所有配置
# 结构: data[op] = [{"working_on":..., "results":..., "failed":True/False}, ...]
data = defaultdict(list)

last_working = None
last_working_op = None

working_pat = re.compile(r'working on ([^,]+),')
results_pat = re.compile(r'^\[results\] ([^:]+):')

with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")

        m = working_pat.search(line)
        if m:
            last_working = line
            last_working_op = m.group(1)
            continue

        m = results_pat.match(line)
        if m:
            op = m.group(1)
            if op in bad_ops:
                failed = "10000000" in line
                data[op].append({
                    "working_on": last_working if last_working is not None else "(missing working on)",
                    "results": line,
                    "failed": failed,
                })

# 生成 HTML
parts = []
parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Failed Op Report</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 24px;
    line-height: 1.4;
}
h1 {
    margin-bottom: 8px;
}
h2 {
    margin-top: 28px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
}
.summary {
    margin-bottom: 24px;
    color: #444;
}
.entry {
    padding: 10px 12px;
    margin: 10px 0;
    border-radius: 6px;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    font-family: Consolas, Menlo, monospace;
    font-size: 13px;
}
.failed {
    background: #ffe8e8;
    border-color: #e57373;
}
.success {
    background: #eaffea;
    border-color: #81c784;
}
.tag {
    display: inline-block;
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 8px;
    font-family: Arial, sans-serif;
}
.tag-failed {
    background: #d32f2f;
    color: white;
}
.tag-success {
    background: #2e7d32;
    color: white;
}
.counts {
    color: #555;
    margin-bottom: 10px;
}
</style>
</head>
<body>
<h1>Failed Operator Configuration Report</h1>
<div class="summary">This report lists all operators that failed at least once, and shows all observed configurations from the full log. Failed configurations are highlighted in red; successful ones are highlighted in green.</div>
""")

for op in sorted(data.keys()):
    entries = data[op]
    failed_count = sum(1 for x in entries if x["failed"])
    success_count = sum(1 for x in entries if not x["failed"])

    parts.append(f"<h2>{html.escape(op)}</h2>")
    parts.append(
        f'<div class="counts">total={len(entries)}, failed={failed_count}, success={success_count}</div>'
    )

    for item in entries:
        cls = "failed" if item["failed"] else "success"
        tag_cls = "tag-failed" if item["failed"] else "tag-success"
        tag_text = "FAILED" if item["failed"] else "SUCCESS"

        working_escaped = html.escape(item["working_on"])
        results_escaped = html.escape(item["results"])

        parts.append(
            f'<div class="entry {cls}">'
            f'<div class="tag {tag_cls}">{tag_text}</div>\n'
            f'{working_escaped}\n{results_escaped}'
            f'</div>'
        )

parts.append("</body></html>")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

print(f"Wrote report to {OUT_FILE}")