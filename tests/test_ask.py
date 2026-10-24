"""計測の境界条件を、ネットワークとAPIキーなしで確認する。"""

from types import SimpleNamespace

from google.genai import types
import pytest

from app import ask


@pytest.mark.parametrize("usage, expected", [
    (types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12, candidates_token_count=7,
        thoughts_token_count=3, total_token_count=22), (12, 7, 3, 22)),
    (None, (None, None, None, None)),
])
def test_measure_preserves_usage_and_times_only_the_call(monkeypatch, usage, expected):
    times = iter([10.0, 10.25])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(times))
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text="確認用の回答")]),
            finish_reason=types.FinishReason.STOP)],
        usage_metadata=usage, model_version="test-version",
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return response

    result = ask.measure(SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                         "質問", "test-model")
    assert result.elapsed_seconds == 0.25
    assert (result.prompt_tokens, result.output_tokens,
            result.thought_tokens, result.total_tokens) == expected
    assert result.text == "確認用の回答"
    assert result.model_version == "test-version"
    assert result.finish_reason == "STOP"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["contents"] == "質問"
    assert calls[0]["config"].max_output_tokens == 768


def test_benchmark_keeps_failed_attempts_without_raw_error_or_key(monkeypatch, tmp_path):
    from scripts import benchmark_ask as bench
    from google.genai import errors
    import json

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench, "create_client", Client)
    monkeypatch.setattr(bench, "questions", lambda: [("sample", "質問")])

    def fail(*args):
        raise errors.ClientError(429, {"error": {"message": "private-key-value"}})

    monkeypatch.setattr(bench, "measure", fail)
    output = tmp_path / "result.jsonl"
    bench.run(["test-model"], 1, 0, output)
    saved = output.read_text(encoding="utf-8")
    row = json.loads(saved)
    assert row["status"] == "api_error"
    assert row["error_code"] == 429
    assert "private-key-value" not in saved
    assert "output_tokens" not in row  # 失敗を0トークンと数えない
    with pytest.raises(FileExistsError):
        bench.run(["test-model"], 1, 0, output)


def test_summary_excludes_failures_and_does_not_turn_missing_tokens_into_zero():
    from scripts.summarize_benchmark import summarize

    rows = [
        dict(model="m", question_id="q", status="ok", elapsed_seconds=2,
             prompt_tokens=10, output_tokens=20),
        dict(model="m", question_id="q", status="ok", elapsed_seconds=4,
             prompt_tokens=10, output_tokens=None),
        dict(model="m", question_id="q", status="api_error", elapsed_seconds=100),
    ]
    result, = summarize(rows)
    assert result["attempts"] == 3
    assert result["successes"] == 2
    assert result["failures"] == 1
    assert result["elapsed_seconds_median"] == 3
    assert result["seconds_min"] == 2 and result["seconds_max"] == 4
    assert result["output_tokens_median"] == 20
    assert result["thought_tokens_median"] is None
