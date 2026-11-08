# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
- `--trace` で見えたツール呼び出し（ツール名と引数）：
- 1つの質問での LLM 呼び出し回数（`requests`）：
- 追加したツールと、docstring に書いた説明：

## 宿題

発展（a）。`app/agent.py` の `INSTRUCTIONS` と検索・講義要約の2関数をそのまま読み込んだ。Pydantic AI では `Agent(instructions=...)` を作ってから `tool_plain` で登録する。ADK では `Agent(instruction=..., tools=[...])` を `root_agent` として公開する。変更したのは組み立て部分だけ。

Codespaces で ADK 2.14.0 を使用。両方とも `gemini-3.5-flash-lite` で、各質問は新しい実行として試した。講座の既存テストは11件すべて通った。

| 質問 | Pydantic AI（`--trace`） | ADK（`adk run`） |
| --- | --- | --- |
| LoRAとQLoRAの違いは？ | `search_glossary` を `LoRA` と `QLoRA` で1回ずつ呼んだ。モデル呼び出しは2回。用語集を参照したことを明記し、それぞれを説明して第2回と結んだ。 | 最初に「ベースモデルの重みを量子化するか」を示してから、両者を説明して第2回と結んだ。 |
| 第3回では何を学びますか？ | `get_lecture_summary(lecture=3)` を1回呼んだ。モデル呼び出しは2回。ループの応答時間・料金・レート制限にも触れた。 | 同じく「モデル＋ループ＋ツール」を挙げ、PiやClaude Codeをハーネスの例として列挙した。最後に第3回を示した。 |

LoRA の質問では、ADK 版は最初の一文だけで違いが分かった。一方、講義内容の質問では、Pydantic AI 版にあったコストの話が ADK 版には出なかった。2問を1回ずつ試したので、違いがフレームワークによるものかは切り分けられない。ADK の標準 CLI では引数や回数が同じ形では表示されなかったため、表のツール回数は Pydantic AI 側で観察した値だけを載せた。

```bash
uv run --frozen python -m app.agent --trace "LoRAとQLoRAの違いは？"
PYTHONPATH=. GOOGLE_API_KEY="$GEMINI_API_KEY" GOOGLE_GENAI_USE_VERTEXAI=false \
  uv run --frozen --with google-adk==2.14.0 adk run adk_agent/glossary_agent
```
