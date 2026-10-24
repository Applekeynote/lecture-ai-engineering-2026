"""第1回 発展: 2モデル・3条件を同じ設定で測る。APIキーは環境変数のみ。

uv run python -m scripts.benchmark_ask --output lectures/01/results/run.jsonl
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import monotonic, perf_counter, sleep

from google.genai import errors
import httpx

from app.ask import DEFAULT_QUESTION, create_client, measure
from app.config import api_error_message

ROOT = Path(__file__).resolve().parents[1]


def questions() -> list[tuple[str, str]]:
    glossary = json.loads((ROOT / "data/glossary.json").read_text(encoding="utf-8"))
    definition = next(t["definition"] for t in glossary["terms"]
                      if t["id"] == "homework-levels")
    return [
        ("lora", "LoRAとは何ですか？ 日本語200字以内で説明してください。"),
        ("course_no_context", DEFAULT_QUESTION + " 日本語200字以内で説明してください。"),
        ("course_with_context", "次の用語集の抜粋に基づいて質問に答えてください。\n"
         + definition + "\n質問：" + DEFAULT_QUESTION
         + " 日本語200字以内で説明してください。"),
    ]


def run(models: list[str], repeats: int, interval: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # 既存の実測値をうっかり上書きしない。
    with output.open("x", encoding="utf-8") as log, create_client() as client:
        next_start = 0.0
        for repeat in range(1, repeats + 1):
            for question_id, question in questions():
                # モデル順を交互にして、常に同じモデルが先になるのを避ける。
                order = models if repeat % 2 else models[::-1]
                for model in order:
                    sleep(max(0, next_start - monotonic()))
                    next_start = monotonic() + interval
                    row = {
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model": model, "question_id": question_id,
                        "question": question, "repeat": repeat,
                        "python": platform.python_version(),
                        "config": {"max_output_tokens": 768, "sdk_attempts": 1,
                                   "timeout_ms": 120000, "interval_seconds": interval},
                    }
                    start = perf_counter()
                    try:
                        result = measure(client, question, model)
                        row.update(asdict(result))
                        row["status"] = "ok" if result.text else "empty"
                    except errors.APIError as exc:
                        # リクエストやキーを含みうる生の例外は保存しない。
                        row.update(status="api_error", error_code=exc.code,
                                   error=api_error_message(exc.code),
                                   elapsed_seconds=perf_counter() - start)
                    except (httpx.HTTPError, TimeoutError) as exc:
                        row.update(status="transport_error", error=type(exc).__name__,
                                   elapsed_seconds=perf_counter() - start)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"{repeat}/{repeats} {model} {question_id}: "
                          f"{row['status']} {row['elapsed_seconds']:.3f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.interval < 0:
        parser.error("repeatsは1以上、intervalは0以上にしてください")
    run(args.models, args.repeats, args.interval, args.output)


if __name__ == "__main__":
    main()
