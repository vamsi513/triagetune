# TriageTune

TriageTune is a local-first project for adapting a small, openly licensed language model to classify banking-support requests and return structured routing decisions.

The project is developed in checkpoints. Measurements are recorded only after the corresponding command has been run.

## Current status

Phase 1 is complete. The Phase 2 majority-class and TF-IDF validation baselines are complete. Unchanged-model evaluation, adapter training, final test evaluation, and serving have not started.

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

## Routing outputs

The intended inference response contains:

- `category`: a canonical BANKING77 intent
- `priority`: a project-created routing annotation
- `team`: a project-created routing annotation

BANKING77 supplies only the intent. Priority and team mappings require separate approval and will be documented as project-created annotations rather than original dataset labels.

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
│   ├── evaluate.py
│   ├── inference.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   └── train_lora.py
├── tests/
├── Dockerfile
├── README.md
└── requirements.txt
```

## Cost constraint

Development and evaluation must use local hardware or free notebook compute. The planned model range is approximately 0.5B–2B parameters; larger models and paid infrastructure are outside the project scope.
