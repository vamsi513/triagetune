# Provisional routing and unknown-request design

## Scope and provenance

This is a project-created policy proposal, not a BANKING77 annotation or measured production behavior. The dataset supplies 77 categories but no urgency or receiving-team labels. The routing code assigns every approved category exactly one team and one priority. The service marks each route `provisional_project_mapping`. A changed category list causes startup to fail until the mapping is updated.

The policy does not inspect ticket text after classification. A wrong category can therefore cause a wrong route, and urgent language in a request does not independently raise priority. No service-level or human-escalation commitment is implied.

## Team assignment

| Team | Categories | Intended queue |
| --- | ---: | --- |
| `card_support` | 22 | Card access, issuance, and functionality |
| `payments` | 11 | Card payments, direct debits, and refunds |
| `transfers` | 11 | Bank transfers and recipient issues |
| `cash_atm` | 8 | ATM and cash transactions |
| `account_identity` | 11 | Account access, identity, and eligibility |
| `top_up` | 10 | Adding money to an account |
| `foreign_exchange` | 4 | Currency exchange and supported currencies |

The exact category membership is the `TEAM_CATEGORIES` mapping in `src/routing.py`; its 77 labels are checked against the saved test report at service startup and in automated tests.

## Priority assignment

`urgent` (8 categories) is proposed for a potentially compromised account or disputed activity: `card_payment_not_recognised`, `cash_withdrawal_not_recognised`, `compromised_card`, `direct_debit_payment_not_recognised`, `extra_charge_on_statement`, `lost_or_stolen_card`, `lost_or_stolen_phone`, and `transaction_charged_twice`.

`standard` (26 categories) covers blocked access, failed or missing movements, and time-sensitive service problems. The exact set is `STANDARD_CATEGORIES` in `src/routing.py`. The remaining 43 categories are `low`, mainly informational or routine requests. These are proposed relative queue labels, not contractual response times. In particular, any production fraud or hardship escalation needs a separately designed text-level safety rule and human review.

## Unknown-request check set

`data/synthetic/unknown_request_probes.jsonl` contains 32 clearly synthetic, manually specified requests: eight direct in-scope, eight paraphrased in-scope, eight unrelated out-of-scope, and eight financially adjacent out-of-scope. In-scope rows specify an expected BANKING77 category. Out-of-scope rows explicitly have no valid category. These examples are a diagnostic seed, not a representative sample of customers or production traffic.

The earlier 32-probe robustness run found zero safe rejections among eight out-of-scope examples. A malformed or disallowed model category is rejected by the endpoint, but that behavior is not a dependable unknown-request detector: the model often forces out-of-scope requests into a valid banking category.

## Proposed measurement before enabling rejection

1. Freeze and version an evaluation set before selecting a rejection threshold. Keep calibration examples separate from the final held-out checks, and include both clearly unrelated and banking-adjacent unknowns, plus valid but unusual banking requests.
2. Evaluate candidate signals on saved raw outputs or new local inference runs: category validity, token-level confidence or score margin if available, and an independently trained in-scope detector only if there is sufficient labeled data. Do not treat valid JSON as proof of scope.
3. Report the confusion counts for accept/reject decisions: correctly rejected unknowns, accepted unknowns, wrongly rejected known requests, and correctly accepted known requests. Break them down by probe group. Also report category accuracy among accepted known requests, latency, and memory impact.
4. Choose a threshold using calibration data only, then evaluate once on untouched held-out data. Do not enable rejection for untrusted traffic solely on this 32-row synthetic seed. Require broader, independently reviewed examples and an explicit operational tolerance for missed unknowns versus false rejection.

The subsequent exploratory measurement is recorded in `reports/unknown_rejection_evaluation.json` and `reports/unknown_rejection_test_audit.json`. It did not establish a safe threshold; rejection remains disabled in the service.
