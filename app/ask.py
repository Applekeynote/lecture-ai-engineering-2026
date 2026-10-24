"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from dataclasses import asdict, dataclass
import json
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class Measurement:
    text: str
    elapsed_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    cached_tokens: int | None
    total_tokens: int | None
    model_version: str | None
    finish_reason: str | None


def create_client() -> genai.Client:
    # 自動再試行の待ち時間を、1リクエストの応答時間に混ぜない。
    return genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def measure(client: genai.Client, question: str, model: str = MODEL,
            max_output_tokens: int = 768) -> Measurement:
    start = perf_counter()
    response = client.models.generate_content(
        model=model, contents=question,
        config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
    )
    elapsed = perf_counter() - start
    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    finish = candidate.finish_reason if candidate else None
    return Measurement(
        text=response.text or "",
        elapsed_seconds=elapsed,
        prompt_tokens=usage.prompt_token_count if usage else None,
        output_tokens=usage.candidates_token_count if usage else None,
        thought_tokens=usage.thoughts_token_count if usage else None,
        cached_tokens=usage.cached_content_token_count if usage else None,
        total_tokens=usage.total_token_count if usage else None,
        model_version=response.model_version,
        finish_reason=finish.value if finish else None,
    )


def ask(question: str) -> str:
    with create_client() as client:
        result = measure(client, question)
    print(f"[tokens] input={result.prompt_tokens} output={result.output_tokens} "
          f"thought={result.thought_tokens} total={result.total_tokens}")
    print(f"[seconds] {result.elapsed_seconds:.3f}")
    return result.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--json", action="store_true", help="計測値と回答をJSONで表示")
    args = parser.parse_args()

    try:
        if args.json:
            with create_client() as client:
                result = measure(client, args.question)
            print(json.dumps({"model": MODEL, "question": args.question,
                              **asdict(result)}, ensure_ascii=False))
        else:
            print(f"[model] {MODEL}")
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
