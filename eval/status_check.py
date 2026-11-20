#!/usr/bin/env python3
"""One source of truth for statuses: experiments/HYPOTHESES.md, the `| вердикты |` line of each EXP record and the
header of each work/T-*.md. Generates work/STATUS.md and the status block of reports/PROJECT_REPORT.md.

  python3 eval/status_check.py --write   # regenerate after editing statuses
  python3 eval/status_check.py --check   # CI (tests/test_status_sync.py): exit 1 on any divergence
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HYP = ROOT / "experiments" / "HYPOTHESES.md"
STATUS = ROOT / "work" / "STATUS.md"
REPORT = ROOT / "reports" / "PROJECT_REPORT.md"
BEGIN, END = "<!-- status:begin (eval/status_check.py --write) -->", "<!-- status:end -->"

CHECKED = ("accepted", "refuted", "mixed", "partial", "inconclusive")
OPEN = ("waiting", "blocked", "proposed", "planned")
H_TOKENS = CHECKED + OPEN
EXP_TOKENS = ("done", "in progress", "waiting", "blocked", "planned")
T_TOKENS = ("done", "review", "partial", "waiting", "blocked", "todo")


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def field(text, name):
    m = re.search(rf"^\| {name} \|(.*)\|\s*$", text, re.M)
    return m.group(1).strip() if m else None


def load(root=ROOT):
    hyp = (root / "experiments" / "HYPOTHESES.md").read_text()
    main, final = {}, {}
    part = hyp.split("## Итог и решение по каждой гипотезе", 1)
    for line in part[0].splitlines():
        if re.match(r"\| H-\d\d \|", line):
            c = cells(line)
            m = re.match(r"`(\w+)`", c[5])
            main[c[0]] = m.group(1) if m else c[5][:30]
    if len(part) == 2:
        for line in part[1].split("\n## ", 1)[0].splitlines():
            if re.match(r"\| H-\d\d \|", line):
                c = cells(line)
                final[c[0]] = {"status": c[1], "exp": [e.strip() for e in c[2].split(",") if e.strip()], "result": c[3], "decision": c[4]}
    exps = {}
    for f in sorted((root / "experiments").glob("EXP-*.md")):
        t = f.read_text()
        st = field(t, "статус") or ""
        verdicts = re.findall(r"(H-\d\d) (\w+)", field(t, "вердикты") or "")
        exps[f.name[:7]] = {"status": st, "verdicts": verdicts, "title": t.splitlines()[0].lstrip("# ").strip()}
    tasks = {}
    for f in sorted((root / "work").glob("T-0*.md")):
        t = f.read_text()
        tasks[f.name[:5]] = {"status": field(t, "статус") or "", "exp": [e.strip() for e in (field(t, "эксперименты") or "").split(",") if e.strip()],
                             "cond": field(t, "условие"), "title": t.splitlines()[0].lstrip("# ").strip(), "file": f.name}
    return main, final, exps, tasks


def problems(root=ROOT):
    main, final, exps, tasks = load(root)
    out = []
    for e, x in exps.items():
        if not any(x["status"].startswith(tok) for tok in EXP_TOKENS):
            out.append(f"{e}: статус «{x['status'][:40]}» не начинается с {EXP_TOKENS}")
        for h, v in x["verdicts"]:
            if v not in H_TOKENS:
                out.append(f"{e}: вердикт {h} {v} не из {H_TOKENS}")
            if v in CHECKED and not x["status"].startswith(("done", "in progress")):
                out.append(f"{e}: вердикт {h} {v}, а статус записи «{x['status'][:30]}»")
    latest = {}
    for e in sorted(exps):
        for h, v in exps[e]["verdicts"]:
            latest[h] = (e, v)
    for h in sorted(set(main) | set(final)):
        tok = main.get(h)
        if tok not in H_TOKENS:
            out.append(f"{h}: статус в таблице гипотез «{tok}» не из {H_TOKENS}")
        f = final.get(h)
        if f is None:
            out.append(f"{h}: нет строки в «Итог и решение по каждой гипотезе»")
            continue
        if f["status"] != tok:
            out.append(f"{h}: в таблице гипотез {tok}, в итогах {f['status']}")
        for e in f["exp"]:
            if e.startswith("EXP-") and e not in exps:
                out.append(f"{h}: ссылка на несуществующую {e}")
        if h in latest:
            e, v = latest[h]
            if v != tok:
                out.append(f"{h}: последний вердикт {e} говорит {v}, а в гипотезах {tok}")
            if e not in f["exp"]:
                out.append(f"{h}: {e} с вердиктом нет в колонке EXP итогов")
        elif tok in CHECKED:
            out.append(f"{h}: статус {tok}, но ни в одной EXP нет строки «вердикты» с {h}")
        if tok in CHECKED:
            if not re.search(r"\d", f["result"]):
                out.append(f"{h}: проверена, но вывод без чисел")
            if not f["decision"]:
                out.append(f"{h}: проверена, но нет решения для приложения")
        elif not f["decision"]:
            out.append(f"{h}: не проверена и нет условия закрытия")
    for t, x in tasks.items():
        if x["status"] not in T_TOKENS:
            out.append(f"{t}: статус «{x['status']}» не из {T_TOKENS}")
        if x["cond"] is None:
            out.append(f"{t}: нет строки «условие»")
        elif x["status"] not in ("done", "todo") and not x["cond"]:
            out.append(f"{t}: статус {x['status']}, а условие пустое")
        for e in x["exp"]:
            if e not in exps:
                out.append(f"{t}: ссылка на несуществующую {e}")
            elif x["status"] == "done" and exps[e]["status"].startswith("planned"):
                out.append(f"{t}: done, но {e} ещё planned")
    return out


def counts(d, tokens):
    return ", ".join(f"{tok} {sum(v == tok for v in d)}" for tok in tokens if any(v == tok for v in d))


def status_md(root=ROOT):
    main, final, exps, tasks = load(root)
    lines = ["# Статус гипотез и задач", "",
             "Файл генерирует `python3 eval/status_check.py --write` из `experiments/HYPOTHESES.md` (таблица итогов), строк `| вердикты |` в `experiments/EXP-*.md` и шапок `work/T-*.md`. Руками не править, CI падает, если он разошёлся с источниками (`tests/test_status_sync.py`).", "",
             f"Гипотезы: {counts(main.values(), H_TOKENS)}. Задачи: {counts([t['status'] for t in tasks.values()], T_TOKENS)}.", "",
             "## Гипотезы", "| id | статус | где | вывод | решение для приложения или что нужно, чтобы закрыть |", "|---|---|---|---|---|"]
    for h in sorted(final):
        f = final[h]
        st = f"**{f['status']}**" if f["status"] in CHECKED else f["status"]
        lines.append(f"| {h} | {st} | {', '.join(f['exp']) or '–'} | {f['result'] or '–'} | {f['decision'] or '–'} |")
    lines += ["", "## Задачи", "| id | статус | эксперименты | что осталось |", "|---|---|---|---|"]
    for t, x in tasks.items():
        lines.append(f"| [{t}]({x['file']}) | {x['status']} | {', '.join(x['exp']) or '–'} | {x['cond'] or '–'} |")
    return "\n".join(lines) + "\n"


def report_block(root=ROOT):
    main, final, exps, tasks = load(root)
    rows = ["| id | статус | вывод | что меняет в приложении |", "|---|---|---|---|"]
    for h in sorted(final):
        f = final[h]
        if f["status"] in CHECKED:
            rows.append(f"| {h} | {f['status']} | {f['result']} | {f['decision']} |")
    wait = [h for h in sorted(final) if final[h]["status"] in OPEN]
    return "\n".join([BEGIN, f"Проверено {sum(final[h]['status'] in CHECKED for h in final)} гипотез из {len(final)}: {counts(main.values(), CHECKED)}. "
                      f"Не проверены ({counts(main.values(), OPEN)}): {', '.join(wait)}, условия в [`work/STATUS.md`](../work/STATUS.md).", "", *rows, END])


def report_text(root=ROOT):
    t = (root / "reports" / "PROJECT_REPORT.md").read_text()
    if BEGIN not in t:
        return t
    a, rest = t.split(BEGIN, 1)
    return a + report_block(root) + rest.split(END, 1)[1]


def stale(root=ROOT):
    out = []
    if (root / "work" / "STATUS.md").read_text() != status_md(root):
        out.append("work/STATUS.md устарел: python3 eval/status_check.py --write")
    t = (root / "reports" / "PROJECT_REPORT.md").read_text()
    if BEGIN not in t:
        out.append("reports/PROJECT_REPORT.md: нет блока статусов")
    elif t != report_text(root):
        out.append("reports/PROJECT_REPORT.md: блок статусов устарел: python3 eval/status_check.py --write")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    errs = problems()
    if a.write and not errs:
        STATUS.write_text(status_md())
        REPORT.write_text(report_text())
    errs += stale()
    for e in errs:
        print(e)
    print(f"status_check: {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
