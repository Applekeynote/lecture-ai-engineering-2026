"""JSONLの実測ログから、成功数と中央値・範囲を再計算する（API呼び出しなし）。"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["question_id"]].append(row)
    result = []
    for (model, question), attempts in sorted(groups.items()):
        successful = [r for r in attempts if r["status"] == "ok"]
        summary = dict(model=model, question_id=question, attempts=len(attempts),
                       successes=len(successful), failures=len(attempts) - len(successful))
        lengths = [len(r.get("text", "")) for r in successful]
        summary["response_chars_median"] = median(lengths) if lengths else None
        summary["within_200_chars"] = sum(n <= 200 for n in lengths)
        for field in ("elapsed_seconds", "prompt_tokens", "output_tokens", "thought_tokens",
                      "cached_tokens", "total_tokens"):
            values = [r[field] for r in successful if r.get(field) is not None]
            summary[field + "_median"] = median(values) if values else None
            if field == "elapsed_seconds":
                summary["seconds_min"] = min(values) if values else None
                summary["seconds_max"] = max(values) if values else None
        result.append(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.inputs
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = summarize(rows)
    if not summaries:
        parser.error("計測行がありません")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# 回答と計測値（JSONLから生成）", "",
             "APIの回答は原文。欠測は `None`。失敗を成功時の中央値に含めていない。", ""]
    for i, row in enumerate(rows, 1):
        lines += [f"## {i}. {row['model']} / {row['question_id']} / {row['repeat']}回目", "",
                  f"時刻: {row['started_at_utc']} / 結果: {row['status']} / "
                  f"時間: {row['elapsed_seconds']:.3f}秒", "",
                  f"入力: {row.get('prompt_tokens')} / 出力: {row.get('output_tokens')} / "
                  f"思考: {row.get('thought_tokens')} / 合計: {row.get('total_tokens')} / "
                  f"終了理由: {row.get('finish_reason')}", "",
                  f"回答原文の文字数（改行・装飾・文字数の自己申告も含む）: {len(row.get('text', ''))}", "",
                  "質問:", "", "> " + row["question"].replace("\n", "\n> "), "",
                  "回答:", "", "> " + row.get("text", row.get("error", "")).replace("\n", "\n> "), ""]
    (args.output / "responses.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
