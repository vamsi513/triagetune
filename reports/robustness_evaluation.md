# Robustness evaluation

This is a 32-record project-created synthetic probe set, not a production benchmark. It contains eight short, eight long, eight ambiguous, and eight out-of-scope inputs.

| Input group | TF-IDF result | Saved adapter result |
| --- | ---: | ---: |
| Short (exact-match rate) | 0.875 | 1.000 |
| Long (exact-match rate) | 0.750 | 0.875 |
| Ambiguous (plausible-category rate) | 0.875 | 1.000 |
| Out Of Scope (safe-rejection rate) | 0.000 | 0.000 |

Out-of-scope inputs have no valid target among the 77 categories. Both approaches safely rejected 0 of 8 requests. The classical baseline forced all eight into known categories; the adapter forced seven and invented one disallowed category. Neither behavior is counted as a safe rejection.

## Actual failures

### TF-IDF + logistic regression

- `short_03` (short): predicted `pending_cash_withdrawal` for “cash withdrawal declined”
- `long_05` (long): predicted `transfer_not_received_by_recipient` for “I ordered a physical card after opening the account and received a message saying it had been dispatched. The estimated delivery window ended several days ago. I checked the mailbox, asked other people at the address, confirmed that the delivery address in my profile is correct, and still cannot find the envelope. I am not asking how long a new order normally takes; I am trying to locate a card that was already sent but has not arrived.”
- `long_06` (long): predicted `why_verify_identity` for “The app is asking me to prove my identity before I can finish setting up the account. I have tried taking clear photographs in good lighting, made sure the identity document is current, and checked that my personal information matches it exactly. The verification attempt still fails each time. I understand why identity checks may be required, but my problem is that I cannot successfully complete the verification process with the documents I have.”
- `ambiguous_02` (ambiguous): predicted `verify_source_of_funds` for “Where is my money?”
- `out_of_scope_01` (out_of_scope): predicted `fiat_currency_support` for “Can you recommend a pizza place near the train station?”
- `out_of_scope_02` (out_of_scope): predicted `transfer_timing` for “What will the weather be tomorrow afternoon?”
- `out_of_scope_03` (out_of_scope): predicted `passcode_forgotten` for “Please reset the password for my email account.”
- `out_of_scope_04` (out_of_scope): predicted `card_linking` for “How do I repair a leaking kitchen faucet?”
- `out_of_scope_05` (out_of_scope): predicted `card_arrival` for “Which shares should I buy this week?”
- `out_of_scope_06` (out_of_scope): predicted `verify_source_of_funds` for “Find a flight from Chicago to Seattle next Friday.”
- `out_of_scope_07` (out_of_scope): predicted `cash_withdrawal_charge` for “Play some relaxing music for studying.”
- `out_of_scope_08` (out_of_scope): predicted `age_limit` for “Write a birthday invitation for my daughter.”

### Saved LoRA adapter

- `long_06` (long): predicted `why_verify_identity` for “The app is asking me to prove my identity before I can finish setting up the account. I have tried taking clear photographs in good lighting, made sure the identity document is current, and checked that my personal information matches it exactly. The verification attempt still fails each time. I understand why identity checks may be required, but my problem is that I cannot successfully complete the verification process with the documents I have.”
- `out_of_scope_01` (out_of_scope): predicted `unable_to_verify_identity` for “Can you recommend a pizza place near the train station?”
- `out_of_scope_02` (out_of_scope): predicted `unable_to_verify_identity` for “What will the weather be tomorrow afternoon?”
- `out_of_scope_03` (out_of_scope): predicted `why_verify_identity` for “Please reset the password for my email account.”
- `out_of_scope_04` (out_of_scope): predicted `repair_kitchen_faucet` for “How do I repair a leaking kitchen faucet?”
- `out_of_scope_05` (out_of_scope): predicted `why_verify_identity` for “Which shares should I buy this week?”
- `out_of_scope_06` (out_of_scope): predicted `unable_to_verify_identity` for “Find a flight from Chicago to Seattle next Friday.”
- `out_of_scope_07` (out_of_scope): predicted `get_physical_card` for “Play some relaxing music for studying.”
- `out_of_scope_08` (out_of_scope): predicted `get_physical_card` for “Write a birthday invitation for my daughter.”
