# TriageTune

TriageTune is a local-first project for adapting a small, openly licensed language model to classify banking-support requests and return structured routing decisions.

The project is developed in checkpoints. Measurements are recorded only after the corresponding command has been run.

## Current status

Phases 1 through 5 are complete for local use. The saved adapter serves real category predictions through a validated HTTP endpoint, and the CPU-only container built, started, and returned a real prediction. A later provisional routing policy adds project-created priority and team values. These values are deterministic rules based on the predicted category, not dataset labels or separately trained predictions. Unknown-request rejection remains unimplemented and the service is not ready for untrusted traffic.

## Dataset

TriageTune uses BANKING77, a collection of English online-banking queries annotated with 77 fine-grained intents.

- Source: <https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data>
- Dataset license: CC BY 4.0
- Published records: 13,083
- Published training records: 10,003
- Published test records: 3,080
- Provenance: annotated customer queries; not synthetic data

The published test set remains reserved for final evaluation. Only the published training pool is divided into project training and validation splits.

## Measured data-preparation results

The preparation run used seed `42`, a 20% validation allocation from the cleaned published training pool, and a near-duplicate threshold of `0.90`.

| Split | Records | Classes |
| --- | ---: | ---: |
| Train | 7,994 | 77 |
| Validation | 1,998 | 77 |
| Test | 3,080 | 77 |

The source contained no byte-for-byte duplicate query records. Canonical normalization identified 12 equivalent duplicate records caused by case or whitespace differences. Eleven were removed from the training pool. One equivalent pair remains within the published test set so all official test records are preserved. No canonical duplicate crosses the prepared splits.

The similarity audit flagged 422 cross-split near-duplicate pairs: 116 between training and validation, 246 between training and test, and 60 between validation and test. These are audit flags, not automatically removed records. The complete parameters and counts are in `reports/data_preparation_report.json`.

## Prepare the data

The raw `train.csv` and `test.csv` files must be present in `data/raw/`.

```bash
python src/prepare_data.py
```

The command writes cleaned split files and a near-duplicate report to `data/processed/`, plus the full run report to `reports/`.

## Majority-class baseline

The majority baseline selects its single predicted category from the training split and evaluates unchanged predictions on the validation split. It does not load the reserved test split.

```bash
python src/train_baseline.py --baseline majority
```

The measured validation results are recorded in `reports/majority_baseline_validation.json`.

| Metric | Measured value |
| --- | ---: |
| Accuracy | 0.018519 |
| Macro F1 | 0.000472 |
| Correct predictions | 37 of 1,998 |

The training majority category is `card_payment_fee_charged`, with 150 of 7,994 training records. It occurs 37 times in validation. The baseline predicts this category for every validation request; the other 76 classes therefore receive an F1 of zero.

## TF-IDF and logistic-regression baseline

The classical text baseline uses lowercase word unigrams and bigrams, sublinear term frequency, a minimum document frequency of two, and L2 normalization. A multinomial logistic-regression classifier is fit on the 7,994 training examples without class weighting or validation tuning.

```bash
python src/train_baseline.py --baseline tfidf_logreg
```

Measured on 1,998 validation examples:

| Metric | Measured value |
| --- | ---: |
| Accuracy | 0.852853 |
| Macro F1 | 0.848371 |
| Correct predictions | 1,704 of 1,998 |

The two most frequent directed confusions were `activate_my_card` → `card_arrival` and `exchange_rate` → `card_payment_wrong_exchange_rate`, with five validation examples each. The next was `transfer_fee_charged` → `extra_charge_on_statement`, with four examples. Several three-example confusions occurred among closely related card-delivery, transfer-status, verification, and card-functionality intents.

Complete per-class precision, recall, F1, and support are stored in `reports/tfidf_logreg_validation.json`. The full 77×77 confusion matrix and every validation prediction are stored alongside it. The reserved test split was not loaded.

## Unchanged-model structured-output baseline

The unchanged-model baseline uses `Qwen/Qwen2.5-1.5B-Instruct`, a 1.54B-parameter instruction-tuned model released under Apache 2.0. The run used revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, float16 weights, greedy decoding, a batch size of eight, and at most 32 newly generated tokens. No examples were included in the prompt and no model weights were changed.

The approved output schema for this stage contains only the original dataset target:

```json
{"category":"one_allowed_category"}
```

Raw completions are passed directly to `json.loads`. Markdown fences, explanations, extra text, malformed objects, extra keys, and invented labels are not repaired before measurement. Invalid or disallowed outputs count as incorrect classifications.

Measured on all 1,998 validation examples:

| Metric | Measured value |
| --- | ---: |
| Accuracy | 0.297798 |
| Macro F1 | 0.296327 |
| Correct predictions | 595 of 1,998 |
| Invalid JSON | 326 of 1,998 (0.163163) |
| Invalid schema | 375 of 1,998 (0.187688) |
| Invalid or disallowed category | 405 of 1,998 (0.202703) |

Of the 326 invalid JSON completions, 323 used Markdown code fences and three had other syntax failures. Another 49 parsed as JSON but used the wrong object keys. Thirty more used a structurally valid object with a category outside the allowed label set.

The complete metrics, prompt, runtime configuration, and per-class scores are stored in `reports/base_model_validation.json`. Every raw completion is retained unchanged in `reports/base_model_validation_outputs.jsonl`. The reserved test split was not loaded.

```bash
python src/evaluate.py --batch-size 8
```

## Adapter training

The training entry point applies response-only LoRA tuning to the same pinned 1.5B-parameter base model used by the unchanged-model baseline. It masks all system and user prompt tokens from the loss and supervises only the exact category JSON response.

The approved configuration uses rank 16, alpha 32, dropout 0.05, and the query, key, value, and output attention projections. It uses a micro-batch size of two, eight gradient-accumulation steps, a maximum sequence length of 544, a learning rate of `2e-4`, cosine decay, 30 warmup steps, and at most two epochs. Validation and adapter-only checkpoints occur every 250 optimizer steps. A three-hour wall-clock guard stops training between optimizer steps; an in-progress full validation pass is allowed to finish so its measurement and checkpoint remain valid.

Run the disposable two-step hardware check:

```bash
python src/train_lora.py --smoke-test
```

Start a fresh tracked run:

```bash
python src/train_lora.py
```

The completed run stopped after 251 optimizer steps because of the wall-clock guard. The step-250 checkpoint was selected after evaluation on all 1,998 validation records.

| Metric | Measured value |
| --- | ---: |
| Completed optimizer steps | 251 of 1,000 maximum |
| Completed epochs | 0.502377 |
| Aggregate training loss | 0.126404 |
| Validation loss | 0.065802 |
| Tracked training runtime | 13,264.4 seconds |
| Trainable adapter parameters | 4,358,144 |
| Adapter weight file | 17,462,432 bytes |
| Adapter and tokenizer files | 28,893,841 bytes |

The tracked runtime was 3 hours, 41 minutes because the full validation pass began before the three-hour boundary and was allowed to complete. The adapter contains no complete base-model weight file.

Local experiment data is stored under `mlruns/`, the resumable checkpoint under `checkpoints/`, and the final adapter under `artifacts/lora_adapter/`. Those runtime artifacts are excluded from version control. The reproducible configuration, measured losses, runtime, and adapter file hashes are recorded in `reports/lora_training.json`; the tracked loss history is in `reports/lora_training_history.json`.

## Final test evaluation: Stage 1

The untouched 3,080-record test split was evaluated for the first time after training. All approaches used the same test-file fingerprint. Invalid or disallowed structured outputs count as incorrect; structured-output rates do not apply to the two classical classifiers.

| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | Invalid JSON |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.012987 | 0.000169 | 0.012987 | 0.000333 | N/A |
| TF-IDF + logistic regression | 0.846753 | 0.858564 | 0.846753 | 0.845445 | N/A |
| Unchanged base model | 0.301623 | 0.502504 | 0.301623 | 0.307655 | 0.191883 |
| Saved LoRA adapter | 0.804870 | 0.848211 | 0.804870 | 0.804991 | 0.000000 |

The adapter improves accuracy over the unchanged base model by 0.503247 and macro F1 by 0.497336. It also reduces invalid JSON from 591 outputs to zero. However, the frozen TF-IDF baseline remains strongest at this checkpoint, exceeding the adapter by 0.041883 accuracy and 0.040454 macro F1. The adapter produced eight valid objects whose category was outside the allowed set; these still count as incorrect.

Detailed reports contain all 77 per-class precision, recall, F1, and support values. Each approach also has a complete prediction record and confusion matrix. `reports/final_test_evaluation.json` is the validated combined comparison.

Rebuild the classical test reports and then the validated comparison with:

```bash
python src/evaluate_classical.py
python src/summarize_test_evaluation.py
```

## Final test evaluation: Stage 2

Uncertainty was measured without retraining or generating new predictions. The analysis used 10,000 deterministic stratified bootstrap samples with seed `42`; each replicate sampled 40 records with replacement from every category. Paired comparisons reused identical sampled rows for both approaches.

| Approach | Accuracy with 95% CI | Macro F1 with 95% CI |
| --- | ---: | ---: |
| Majority class | 0.012987 [0.012987, 0.012987] | 0.000333 [0.000333, 0.000333] |
| TF-IDF + logistic regression | 0.846753 [0.834740, 0.858766] | 0.845445 [0.832749, 0.857287] |
| Unchanged base model | 0.301623 [0.288312, 0.314935] | 0.307655 [0.291863, 0.320361] |
| Saved LoRA adapter | 0.804870 [0.792532, 0.817208] | 0.804991 [0.791518, 0.816938] |

The paired TF-IDF-minus-adapter difference is 0.041883 accuracy with a 95% interval of [0.027273, 0.056494], and 0.040454 macro F1 with an interval of [0.025569, 0.055830]. Both intervals exclude zero. The adapter exceeds the unchanged base model by 0.503247 accuracy with an interval of [0.486364, 0.519805].

The adapter has 601 errors. Its largest concentrations are `top_up_failed` → `topping_up_by_card` (29), `top_up_limits` → `topping_up_by_card` (26), `card_swallowed` → `cash_withdrawal_not_recognised` (23), and `pin_blocked` → `change_pin` (16). It beats the TF-IDF baseline on class F1 for 29 categories and trails it on 48. Of the 3,080 examples, both approaches are correct on 2,224, only TF-IDF is correct on 384, only the adapter is correct on 255, and both are wrong on 217.

The complete method, all metric intervals, paired differences, correctness overlaps, weakest categories, top 15 confusion pairs, and representative examples are in `reports/test_error_analysis.json`. A concise readable version is in `reports/test_error_analysis.md`.

Reproduce Stage 2 from the saved predictions with:

```bash
python src/analyze_test_results.py
```

## Final test evaluation: Stage 3

Robustness was checked with 32 project-created synthetic probes: eight short, eight long, eight ambiguous, and eight out-of-scope requests. This is a small diagnostic suite, not a production benchmark. No retraining occurred. Ambiguous probes use documented sets of plausible categories instead of invented single ground-truth labels, and out-of-scope probes have no valid target among the 77 categories.

| Input group | Measurement | TF-IDF | Saved adapter |
| --- | --- | ---: | ---: |
| Short | Exact category | 7/8 | 8/8 |
| Long | Exact category | 6/8 | 7/8 |
| Ambiguous | Category within documented plausible set | 7/8 | 8/8 |
| Out of scope | Safe rejection | 0/8 | 0/8 |

The adapter returned valid JSON with the required key for all 32 probes. It returned an allowed category for 31; for the leaking-faucet request it invented `repair_kitchen_faucet`. On the other seven out-of-scope probes it forced an unrelated banking category. The classical baseline forced all eight out-of-scope requests into known categories. Neither approach currently has safe unknown-request behavior, so an explicit rejection mechanism is required before serving untrusted inputs.

The adapter’s one long-input error mapped a failed identity-verification request to `why_verify_identity` instead of `unable_to_verify_identity`. Its largest padded prompt in this run was 518 tokens. Complete raw predictions, configuration hashes, runtime measurements, and every failure are stored in `reports/robustness_predictions.jsonl`, `reports/robustness_evaluation.json`, and `reports/robustness_evaluation.md`.

Reproduce the diagnostic run with:

```bash
python src/evaluate_robustness.py
```

## Routing outputs

The current response has a canonical BANKING77 `category`, plus project-created `priority` and `team` values derived from that category. `routing_policy` is always `provisional_project_mapping` to identify this distinction. BANKING77 supplies only the intent; it does not provide priority or team ground truth. The exhaustive 77-category mapping and rule checks are in `src/routing.py`. Policy rationale, group counts, and the unknown-request test design are in `reports/routing_rejection_design.md`. These provisional values are not a validated operational escalation policy.

## Exploratory unknown-request rejection

The saved model and two lightweight score signals were measured on 32 clearly synthetic requests: 16 in-scope and 16 out-of-scope. The first four requests in each of four groups formed a 16-row calibration half; the remaining 16 formed a held-out half. No retraining occurred. The saved model assigned an approved banking category to 9 of the 16 out-of-scope requests and exactly matched 13 of 16 in-scope target categories.

| Decision rule | Held-out unknowns rejected | Held-out known requests wrongly rejected |
| --- | ---: | ---: |
| Invalid or disallowed model output only | 4/8 | 0/8 |
| Classical-model maximum-probability threshold | 5/8 | 1/8 |
| Nearest-training-text similarity threshold | 5/8 | 1/8 |

The two thresholds were selected only on the synthetic calibration half. Applied unchanged to the previously evaluated 3,080-record BANKING77 test split, the maximum-probability rule would wrongly reject 85 valid banking requests (2.76%), including 42 that the saved model classified correctly. The similarity rule would wrongly reject 123 (3.99%), including 72 correctly classified requests. These are exploratory results, not production estimates: the synthetic set is small, manually authored, and shares patterns across its halves. Neither rule is enabled in the service.

Reproduce the measurements with the locally saved adapter and classical model:

```bash
.venv/bin/python -m src.evaluate_unknown_rejection
.venv/bin/python -m src.audit_rejection_on_test
```

The row-level synthetic results and complete summaries are stored in `reports/unknown_rejection_predictions.jsonl`, `reports/unknown_rejection_evaluation.json`, and `reports/unknown_rejection_test_audit.json`.

## Local serving: Phase 5

The service loads the saved adapter once at startup and exposes `GET /health`, `GET /model-info`, and `POST /classify`. Classification accepts a nonblank `text` field and returns a validated canonical `category` with the provisional route. Extra request fields, text over 2,000 characters, and rendered prompts over 1,024 tokens are rejected. A malformed or disallowed model category returns HTTP 422; it is never silently repaired or represented as a successful prediction. Generation is serialized behind a lock to limit accelerator-memory pressure under concurrent requests.

Run the native service from the project root after installing `requirements.txt` and ensuring the locally saved adapter and model cache are present:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8765
```

Example request:

```bash
curl -X POST http://127.0.0.1:8765/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"I cannot use my PIN."}'
```

Before the provisional policy was added, this returned `{"category":"pin_blocked"}` during the measured local run. The current endpoint additionally returns `priority`, `team`, and `routing_policy`; its route is determined only after the category passes validation. A separate in-scope card-swallowed request returned HTTP 422 because the model output was disallowed, so successful handling is not universal. The out-of-scope robustness result remains unresolved; this endpoint should not be exposed as a trusted general-purpose router.

The container uses a CPU-only runtime. It deliberately excludes model weights from the image; provide the ignored local adapter and base-model cache as read-only mounts:

```bash
docker build -t triagetune-local:phase5 .
docker run --rm -p 127.0.0.1:8766:8000 \
  -v "$PWD/artifacts/lora_adapter:/models/adapter:ro" \
  -v "$PWD/.cache/huggingface:/models/cache:ro" \
  triagetune-local:phase5
```

The image built and ran on ARM Linux. Health and model-info returned HTTP 200, the container reached healthy status, and the same example returned the expected category. Container loading took 35.54 seconds on CPU and observed memory after inference was 6.942 GiB of a 7.748 GiB Docker limit. This is functional but too memory-heavy for a small laptop to leave running casually.

The native service loaded in 5.21 seconds on the Mac accelerator. A local test used 20 measured requests with four concurrent clients after one warmup: 20/20 returned the same valid category, with 1.46-second P50 and 1.50-second P95 end-to-end latency. Peak process RSS during that run was 225,001,472 bytes; it excludes accelerator allocations. The serialized adapter and tokenizer occupy 28,893,841 bytes, of which 17,462,432 bytes are adapter weights. Full machine-readable measurements are in `reports/serving_load_test.json` and `reports/serving_runtime.json`.

Run the automated checks with:

```bash
pytest -q
```

The recorded Phase 5 run had 16 passing tests. The later provisional-policy changes add checks for exact category coverage, route outputs, synthetic-probe consistency, and rejection-score calculations. The current run passed all 22 tests. The service runs only on loopback in the documented commands; production deployment and unknown-request rejection are not claimed.

## Project layout

```text
triagetune/
├── api/
│   └── main.py
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── reports/
├── src/
│   ├── analyze_test_results.py
│   ├── evaluate.py
│   ├── evaluate_classical.py
│   ├── evaluate_robustness.py
│   ├── inference.py
│   ├── prepare_data.py
│   ├── summarize_test_evaluation.py
│   ├── train_baseline.py
│   └── train_lora.py
├── tests/
├── Dockerfile
├── README.md
└── requirements.txt
```

## Cost constraint

Development and evaluation must use local hardware or free notebook compute. The planned model range is approximately 0.5B–2B parameters; larger models and paid infrastructure are outside the project scope.
