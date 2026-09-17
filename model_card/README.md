---
language: en
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
datasets:
  - PolyAI/banking77
pipeline_tag: text-generation
tags:
  - lora
  - banking-intent-classification
  - structured-output
  - research
---

# TriageTune BANKING77 intent adapter

This LoRA adapter classifies English banking-support requests into one of the 77 BANKING77 intent labels. It returns a JSON object with one `category` field. It is **not** a full copy of the base model; loading it requires the [Qwen2.5-1.5B-Instruct base model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

**License:** this adapter package is licensed under Apache 2.0 (see `LICENSE`). The [project source repository](https://github.com/vamsi513/triagetune) is independently licensed under Apache 2.0 for project-authored code and documentation. The base model is separately distributed under Apache 2.0; the [BANKING77 publisher's dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) is separately under [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). These Apache licenses do not replace the dataset license.

## Intended use

Research and controlled local evaluation of single-label banking intent classification on requests similar to BANKING77. Outputs should be parsed and checked against the accompanying `labels.json`; malformed JSON, extra keys, and disallowed labels must be rejected. A predicted category is not a verified decision about the customer's account or a recommended action.

Do not use this adapter for live or automatic support routing, financial decisions, fraud handling, or unsupervised replies to customers. It has no reliable unsupported-request detector. The project's category-to-team and priority rules are separately authored, provisional annotations; they are **not** learned by this adapter or approved operational policy. Do not send sensitive customer information to a public demonstration endpoint.

## Model and data

- Base: `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Format: PEFT LoRA adapter targeting `q_proj`, `k_proj`, `v_proj`, and `o_proj`; 4,358,144 trainable parameters. Saved adapter weights are 17,462,432 bytes. Full base weights are not included.
- Data: 7,994 training, 1,998 validation, and 3,080 untouched test requests across 77 intent categories. The test split has 40 requests per category; the training split is not balanced. Eleven equivalent duplicates were removed from the original training pool. A further 422 near-duplicate cross-split pairs were flagged but not removed; this is a possible source of optimistic performance. No exact canonical duplicate crosses the prepared splits.
- Training: response-only loss with prompt tokens masked; rank 16, alpha 32, dropout 0.05; maximum sequence length 544; micro-batch 2 with accumulation 8; learning rate 0.0002 with cosine schedule and 30 warmup steps; seed 42. The run stopped at 251 optimizer steps (0.5024 epochs) because of its wall-clock guard. The selected checkpoint was step 250, with validation loss 0.065802. Measured training runtime was 13,264 seconds on Apple MPS in float16, including work after the guard was reached.

## Evaluation

All figures below use the same reserved, 3,080-record BANKING77 test split. Accuracy and macro F1 score invalid or disallowed model outputs as wrong. Generation was greedy with at most 32 new tokens and the exact 77-label prompt shown in the usage example.

| Approach | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority-class baseline | 1.30% | 0.03% |
| Unchanged base model | 30.16% | 30.77% |
| Saved LoRA adapter | 80.49% | 80.50% |
| TF-IDF + logistic regression | 84.68% | 84.54% |

The adapter correctly classified 2,479 of 3,080 requests. It produced zero invalid JSON objects and eight disallowed labels (0.26%); these eight were counted as errors. The unchanged base model had 19.19% invalid JSON. The stronger classical baseline exceeded the adapter by 4.19 percentage points in accuracy and 4.05 points in macro F1 on this test.

Stratified bootstrap intervals (10,000 samples, 95%, 40 records sampled with replacement per category) were [79.25%, 81.72%] for adapter accuracy and [79.15%, 81.69%] for adapter macro F1. Paired intervals for the classical baseline's advantage over the adapter were [2.73, 5.65] percentage points in accuracy and [2.56, 5.58] points in macro F1. These intervals describe sampling uncertainty on this test distribution, not deployment uncertainty.

## Limitations and safety

The model is constrained to a closed label set and often forces unrelated requests into a banking category. On an initial **project-created synthetic** set of 32 probes, it classified 8/8 short and 7/8 long banking examples correctly, gave a plausible label for 8/8 ambiguous examples, and safely rejected 0/8 out-of-scope examples. This small diagnostic is not a real-world performance estimate. A later experimental rejection threshold also failed to establish a safe deployment rule: it would wrongly reject 85 of 3,080 valid test requests. No rejection threshold is enabled in the service.

The evaluation uses English, short written requests drawn from one public benchmark. It does not establish performance for other languages, real customer traffic, changed products or policies, adversarial inputs, demographic subgroups, or privacy-sensitive content. No fairness or privacy audit has been completed. Serving authentication, rate limits, retention rules, and incident handling have not been verified. Human review and a separately validated scope detector are required before any operational use.

## Load and classify locally

The adapter repository contains this card, `LICENSE`, and `labels.json` beside `adapter_config.json`, `adapter_model.safetensors`, and the saved tokenizer files. The example downloads the adapter repository and loads the base model separately. Install compatible `torch`, `transformers`, `peft`, and `huggingface_hub` packages first; loading the base model requires sufficient memory.

```python
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = Path(snapshot_download(repo_id="Vamsi513/triagetune-banking77-lora"))
labels = json.loads((adapter_dir / "labels.json").read_text(encoding="utf-8"))
system_prompt = (
    "Classify one banking support request. Respond with exactly one JSON object "
    'using this schema: {"category":"one_allowed_category"}. '
    "Do not add markdown, explanation, or extra keys. The allowed categories are: "
    f"{', '.join(labels)}"
)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    dtype=torch.float32,  # Portable CPU example; use an appropriate dtype on other devices.
)
model = PeftModel.from_pretrained(base, adapter_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "My card PIN is blocked."},
]
rendered = tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(rendered, return_tensors="pt")
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
raw = tokenizer.decode(
    generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)
parsed = json.loads(raw)
if not isinstance(parsed, dict) or set(parsed) != {"category"}:
    raise ValueError("Invalid output schema")
if not isinstance(parsed["category"], str) or parsed["category"] not in labels:
    raise ValueError("Disallowed category")
print(parsed)
```

The example validates structure but does **not** detect out-of-scope inputs or make the model suitable for production routing.

## Provenance

Saved adapter SHA-256: `3ef2bce78ceaccf108bb2459f32ddedd46933310b0a03a9f93588aa1778a3e29`. Exact system-prompt SHA-256: `2bb904f1e98e5ae0f29b495e9efd4446794e7bfd08424f9b48d04715e3a67039`. The [public source repository](https://github.com/vamsi513/triagetune) contains the training and evaluation reports, including full configuration, per-request outputs, uncertainty calculations, and error analysis. Those reports are not part of this adapter package.
