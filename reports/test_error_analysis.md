# Final test uncertainty and error analysis

Intervals use 10,000 deterministic stratified bootstrap samples. Each sample draws 40 records with replacement from every one of the 77 categories. Paired comparisons reuse the same sampled rows for both approaches.

## Metric intervals

| Approach | Accuracy (95% CI) | Macro F1 (95% CI) |
| --- | ---: | ---: |
| Majority class | 0.012987 [0.012987, 0.012987] | 0.000333 [0.000333, 0.000333] |
| TF-IDF + logistic regression | 0.846753 [0.834740, 0.858766] | 0.845445 [0.832749, 0.857287] |
| Unchanged base model | 0.301623 [0.288312, 0.314935] | 0.307655 [0.291863, 0.320361] |
| Saved LoRA adapter | 0.804870 [0.792532, 0.817208] | 0.804991 [0.791518, 0.816938] |

## Paired comparisons

- TF-IDF + logistic regression minus Saved LoRA adapter: accuracy 0.041883 (95% CI [0.027273, 0.056494]); macro F1 0.040454 (95% CI [0.025569, 0.055830]).
- Saved LoRA adapter minus Unchanged base model: accuracy 0.503247 (95% CI [0.486364, 0.519805]); macro F1 0.497336 (95% CI [0.480740, 0.515882]).
- TF-IDF + logistic regression minus Unchanged base model: accuracy 0.545130 (95% CI [0.528563, 0.562013]); macro F1 0.537790 (95% CI [0.521514, 0.556867]).

## Error concentrations

### TF-IDF + logistic regression

Errors: 472; invalid or disallowed outputs: 0; categories below 0.50 F1: 1.

Most frequent confusion pairs:

- `virtual_card_not_working` → `get_disposable_virtual_card`: 11 records
- `virtual_card_not_working` → `getting_virtual_card`: 10 records
- `card_acceptance` → `card_not_working`: 8 records
- `get_disposable_virtual_card` → `disposable_card_limits`: 6 records
- `atm_support` → `declined_cash_withdrawal`: 5 records

Lowest five class F1 values: `virtual_card_not_working` (0.491), `card_not_working` (0.674), `get_disposable_virtual_card` (0.683), `pending_transfer` (0.706), `balance_not_updated_after_bank_transfer` (0.713).

### Saved LoRA adapter

Errors: 601; invalid or disallowed outputs: 8; categories below 0.50 F1: 4.

Most frequent confusion pairs:

- `top_up_failed` → `topping_up_by_card`: 29 records
- `top_up_limits` → `topping_up_by_card`: 26 records
- `card_swallowed` → `cash_withdrawal_not_recognised`: 23 records
- `pin_blocked` → `change_pin`: 16 records
- `transfer_not_received_by_recipient` → `pending_transfer`: 15 records

Lowest five class F1 values: `top_up_failed` (0.298), `topping_up_by_card` (0.331), `card_swallowed` (0.431), `transfer_not_received_by_recipient` (0.484), `top_up_limits` (0.519).

### Unchanged base model

Errors: 2151; invalid or disallowed outputs: 714; categories below 0.50 F1: 58.

Most frequent confusion pairs:

- `fiat_currency_support` → `__invalid_or_disallowed__`: 35 records
- `getting_virtual_card` → `get_disposable_virtual_card`: 31 records
- `exchange_via_app` → `__invalid_or_disallowed__`: 30 records
- `age_limit` → `__invalid_or_disallowed__`: 28 records
- `supported_cards_and_currencies` → `__invalid_or_disallowed__`: 28 records

Lowest five class F1 values: `card_delivery_estimate` (0.000), `card_payment_wrong_exchange_rate` (0.000), `card_swallowed` (0.000), `exchange_via_app` (0.000), `failed_transfer` (0.000).
