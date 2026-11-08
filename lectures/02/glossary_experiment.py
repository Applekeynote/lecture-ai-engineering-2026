"""Train Qwen LoRA on the lecture glossary and compare it with literal search."""

import argparse
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SYSTEM = (
    "AIE2026の用語集に沿って日本語で答えてください。"
    '出力は {"answer":"回答","needs_review":false} のJSON一行です。'
    "用語集で確認できない場合は、推測せず「用語集に記載がありません。確認が必要です。」と"
    "答え、needs_reviewをtrueにしてください。"
)
UNKNOWN_ANSWER = "用語集に記載がありません。確認が必要です。"
TRAIN_UNKNOWN = [
    "Speculative Decodingの仕組みを教えて。",
    "RLHFの定義を知りたい。",
    "DPOの意味は？",
    "BLEUスコアは何を測りますか。",
    "フラッシュアテンションを説明してください。",
    "GQAとはどんな仕組みですか。",
    "ALiBiの説明をお願いします。",
    "クロスエンコーダとは何をするものですか。",
]
# Each tuple lists acceptable variants for one required concept.
TEST = [
    ("term", "LoRAの重みの更新方法を教えて。", "lora", [("固定", "凍結"), ("低ランク", "小さな行列")]),
    ("term", "過学習とはどんな状態ですか。", "overfitting", [("学習データ", "訓練データ"), ("未知", "汎化", "新しい")]),
    ("term", "コンテキストウィンドウが制限するものは？", "context-window", [("トークン",), ("上限", "最大")]),
    ("term", "エポックはどう数えますか。", "epoch", [("学習データ", "訓練データ"), ("1周", "一周", "1回")]),
    ("paraphrase", "モデルに決まった型のJSONを返させる仕組みは何？", "structured-output", [("型", "スキーマ"), ("JSON", "後処理")]),
    ("paraphrase", "1分あたりに呼び出せる回数を超えたらどうなりますか。", "rate-limit", [("429",), ("上限", "制限")]),
    ("paraphrase", "パラメータを一度にどれくらい動かすかを決める設定は？", "learning-rate", [("学習率",), ("更新",)]),
    ("paraphrase", "温度パラメータを下げると次の語の選び方はどう変わりますか。", "temperature", [("ランダム", "多様", "ばらつき"), ("安定", "減", "低", "決定")]),
    ("missing", "DoRAの更新方法は？", None, []),
    ("missing", "ReActの推論と行動はどうつながりますか。", None, []),
    ("missing", "KVキャッシュは何を保存しますか。", None, []),
    ("missing", "蒸留温度を上げるのはなぜですか。", None, []),
]


def json_answer(answer, needs_review):
    return json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def prepare(repo, out):
    script = repo / "scripts/make_sft_dataset.py"
    raw = subprocess.check_output([sys.executable, "-X", "utf8", str(script)], cwd=repo)
    (out / "glossary_sft.jsonl").write_bytes(raw)
    records = [json.loads(line)["messages"] for line in raw.decode("utf-8").splitlines()]
    train = [
        {"messages": messages(question["content"]) + [
            {"role": "assistant", "content": json_answer(answer["content"], False)}
        ]}
        for question, answer in records
    ]
    train.extend(
        {"messages": messages(question) + [{"role": "assistant", "content": json_answer(UNKNOWN_ANSWER, True)}]}
        for question in TRAIN_UNKNOWN
    )
    assert {r["messages"][1]["content"] for r in train}.isdisjoint(question for _, question, _, _ in TEST)
    (out / "test_gold.json").write_text(
        json.dumps(TEST, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return train


def canonical(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def grade(raw, target, concepts):
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, False, False
    if not (
        "\n" not in raw and "\r" not in raw
        and isinstance(parsed, dict) and set(parsed) == {"answer", "needs_review"}
        and isinstance(parsed["answer"], str) and type(parsed["needs_review"]) is bool
    ):
        return False, False, False
    answer = canonical(parsed["answer"])
    if target is None:
        return True, parsed["needs_review"] and "確認" in answer, not parsed["needs_review"]
    return True, not parsed["needs_review"] and all(
        any(canonical(variant) in answer for variant in group) for group in concepts
    ), False


def evaluate(generate, label, out):
    rows = []
    for category, question, target, concepts in TEST:
        raw = generate(question)
        fmt, content, unsupported = grade(raw, target, concepts)
        rows.append(dict(category=category, question=question, raw=raw, format_ok=fmt,
                         content_ok=content, unsupported=unsupported))
    summary = {}
    for group in ("all", "term", "paraphrase", "missing"):
        subset = [row for row in rows if group == "all" or row["category"] == group]
        summary[group] = dict(n=len(subset), format_ok=sum(r["format_ok"] for r in subset),
                              content_ok=sum(r["content_ok"] for r in subset),
                              unsupported=sum(r["unsupported"] for r in subset))
    result = dict(summary=summary, rows=rows)
    (out / f"{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(label, json.dumps(summary, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from app.tools import load_glossary, search_glossary

    vocabulary = {entry["id"] for entry in load_glossary()}
    assert all(term is None or term in vocabulary for _, _, term, _ in TEST)
    train = prepare(repo, out)
    print(f"Training records: {len(train)}, including {len(TRAIN_UNKNOWN)} abstention examples.", flush=True)

    def lookup(question):
        hits = search_glossary(question)
        if not hits.startswith("["):
            return json_answer(UNKNOWN_ANSWER, True)
        match = json.loads(hits)[0]
        return json_answer(f'{match["term"]}：{match["definition"]}', False)

    search = evaluate(lookup, "search", out)
    if args.search_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before training.")
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to("cuda")

    def generator(target):
        target.eval()
        target.config.use_cache = True

        def generate(question):
            prompt = tok.apply_chat_template(messages(question), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt").to(target.device)
            with torch.inference_mode():
                output = target.generate(**inputs, max_new_tokens=128, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
            return tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return generate

    before = evaluate(generator(model), "before", out)
    model.config.use_cache = False
    # Use only completion tokens for the loss, keeping question length out of the target.
    dataset = Dataset.from_list([
        {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]} for row in train
    ])
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out / "checkpoints"), num_train_epochs=3,
            per_device_train_batch_size=4, gradient_accumulation_steps=2,
            learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=6,
            max_length=384, packing=False, completion_only_loss=True, logging_steps=8,
            save_strategy="no", report_to="none", seed=42,
        ),
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    start = time.monotonic()
    training = trainer.train()
    seconds = round(time.monotonic() - start, 2)
    after = evaluate(generator(trainer.model), "after", out)
    result = dict(model=MODEL, revision=REVISION, seed=42, epochs=3,
                  steps=trainer.state.global_step, seconds=seconds,
                  training_loss=training.training_loss, training_records=len(train),
                  before=before["summary"], after=after["summary"], search=search["summary"])
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT_JSON", json.dumps(result, ensure_ascii=False), flush=True)
    print("DETAIL_JSON", json.dumps({"before": before["rows"], "after": after["rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
