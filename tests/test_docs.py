"""文档一致性检查：防止 README 的数字与评测产物悄悄分家。

这次是真出过事：`results/report.md` 已经是 92 个引用，README 里还写着 89——
两个数字都在同一份仓库里，读者随手一对比就会认为作者没在维护。

只校验**跨轮稳定**的两项（Recall 与作答数），引用条数每轮会浮动（90–94），
把它也钉死会让测试天天红，那就又变成"红久了没人看"。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "results" / "report.md"
DOCS = sorted((ROOT / "docs").glob("*.md"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_placeholder_left_in_docs() -> None:
    """`{X}` 是给人填空的占位符，填完就不该留在交付文档里。"""
    leftovers = [
        f"{path.name}: {line.strip()[:60]}"
        for path in DOCS
        for line in _read(path).splitlines()
        if "{X}" in line
    ]
    assert leftovers == [], f"还有未填的占位符：{leftovers}"


def test_no_todo_in_code_or_docs() -> None:
    """交付前不留 TODO/FIXME：写了就得做完，做不完就写进「已知限制」。

    本文件自己被排除：它必须写出这几个词才能检查它们。
    """
    marker = re.compile(r"\b(TODO|FIXME|XXX)\b")
    targets = [
        path
        for path in list((ROOT / "app").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
        + DOCS
        if path.resolve() != Path(__file__).resolve()
    ]
    hits = [
        f"{path.relative_to(ROOT).as_posix()}: {line.strip()[:60]}"
        for path in targets
        for line in _read(path).splitlines()
        if marker.search(line)
    ]
    assert hits == [], f"还有未处理标记：{hits}"


@pytest.mark.skipif(not REPORT.is_file(), reason="results/report.md 尚未生成（先跑 run_eval）")
def test_readme_recall_matches_report() -> None:
    """README 里 hybrid 的块级 Recall 必须等于报告里的那一格。"""
    report = _read(REPORT)
    match = re.search(r"^\|\s*`hybrid`\s*\|\s*\*\*([\d.]+)\*\*", report, re.MULTILINE)
    if match is None:
        pytest.skip("报告里没有 hybrid 行（格式可能变了）")
    report_value = match.group(1)

    readme = _read(README)
    readme_match = re.search(r"^\|\s*`hybrid`\s*\|\s*\*\*([\d.]+)\*\*", readme, re.MULTILINE)
    assert readme_match is not None, "README 的检索表里找不到 hybrid 行"
    assert readme_match.group(1) == report_value, (
        f"README 写 {readme_match.group(1)}，报告是 {report_value}——改完报告记得同步 README"
    )


@pytest.mark.skipif(not REPORT.is_file(), reason="results/report.md 尚未生成（先跑 run_eval）")
def test_readme_answered_count_matches_report() -> None:
    """README 的「可答题作答」必须等于报告里的作答数。"""
    report = _read(REPORT)
    match = re.search(r"^\|\s*作答\s*\|\s*(\d+)\s*题", report, re.MULTILINE)
    if match is None:
        pytest.skip("报告里没有作答行")
    report_value = match.group(1)

    readme = _read(README)
    readme_match = re.search(r"可答题作答\s*\|\s*\*\*(\d+)\s*/\s*\d+\*\*", readme)
    assert readme_match is not None, "README 的生成表里找不到「可答题作答」行"
    assert readme_match.group(1) == report_value, (
        f"README 写 {readme_match.group(1)} 题，报告是 {report_value} 题"
    )
