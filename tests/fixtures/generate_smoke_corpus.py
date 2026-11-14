"""生成 CI 用的小语料与小题库：**让第三道闸真的能跑，而不是摆着看**。

为什么在 CI 里现生成、而不是把 PDF 提交进仓库：
- 二进制文件不进仓库（也避免「示例文档算不算侵权」的争议）；
- 内容由代码生成，任何人看到这批题就知道它们答的是什么，不需要额外解释。

产物（默认落进仓库自身的 data/ 目录，CI 里跑完就丢）：

    data/raw/ci_smoke.pdf        2 页，一页表格型、一页叙述型
    data/eval/ci_smoke.jsonl     3 道题：2 道可答 + 1 道库里没有

用法（`tests/` 不是包，所以按路径调用）：

    python tests/fixtures/generate_smoke_corpus.py
    python -m app.ingest.pipeline data/raw
    python -m app.retrieval.bench --rebuild
    python tests/check_citations.py --queries data/eval/ci_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGES = [
    [
        "4 接线",
        "4.4 主功率线缆",
        "4.4.1 选型",
        "按表4-2 选择线径，CI-4T22G 机型推荐 16mm 铜芯线。",
        "母线电压低于 380V 时禁止运行。",
    ],
    [
        "6 故障处理",
        "6.2 上电无显示",
        "检查主保险丝与显示排线，若 Err23 持续报出则更换控制板。",
    ],
]

QUESTIONS = [
    {
        "id": "c01",
        "type": "lookup",
        "query": "CI-4T22G 机型的主功率线缆推荐多粗",
        "gold": [{"doc": "ci_smoke.pdf", "page": 1}],
        "expect": "answer",
        "note": "表格型：答案在 4.4.1 选型那段",
    },
    {
        "id": "c02",
        "type": "reasoning",
        "query": "上电无显示，Err23 一直报出来该怎么处理",
        "gold": [{"doc": "ci_smoke.pdf", "page": 2}],
        "expect": "answer",
        "note": "叙述型：保险丝、排线、控制板",
    },
    {
        "id": "c03",
        "type": "unanswerable",
        "query": "这批设备的采购单价和付款方式是什么",
        "gold": [],
        "expect": "refusal",
        "note": "小语料里确实没有价格信息，必须拒答",
    },
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import fitz  # PyMuPDF：仓库本就依赖它

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / "ci_smoke.pdf"

    document = fitz.open()
    for lines in PAGES:
        page = document.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=12, fontname="china-s")
            y += 22.0
    document.save(str(pdf_path))
    document.close()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "ci_smoke.jsonl"
    eval_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in QUESTIONS) + "\n",
        encoding="utf-8",
    )

    print(f"已生成 {pdf_path.as_posix()}（{len(PAGES)} 页）与 {eval_path.as_posix()}（{len(QUESTIONS)} 题）")
    print("下一步：python -m app.ingest.pipeline " + raw_dir.as_posix())
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CI 冒烟语料与题库")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--eval-dir", default="data/eval")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
