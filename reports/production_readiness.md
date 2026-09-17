# Production-readiness assessment

Status: **not ready for untrusted or automatic routing**. This assessment records observed results and release blockers; it does not certify a deployment.

## Measured evidence

- On the 3,080-record BANKING77 test split, the saved adapter reached 80.49% category accuracy. The frozen classical baseline reached 84.68%.
- The original eight synthetic out-of-scope probes received zero safe rejections. A later 32-request synthetic study found that the best score rule rejected five of eight held-out unknowns and wrongly rejected one of eight held-out known requests.
- The same frozen score rule would wrongly reject 85 of 3,080 valid banking test requests (2.76%), including 42 with a correct model category.
- On 600 independently authored external intent requests selected by label, the frozen score threshold rejected 399 and accepted 201. This score-only check did not include model-output validity. In a separate deterministic 40-request model sample, 35 received a banking category and five produced invalid or disallowed categories.
- The CPU container previously returned a real prediction but used about 6.94 GiB after inference. This is not a deployment capacity test.

All external examples were authored for a different application. Label-level selection has not been individually reviewed against this project's 77 categories. None of these test sets represents live traffic.

## Controls now in place

- Provisional category-to-priority/team rules cover all 77 approved categories and are labeled as project-created.
- Response validation rejects malformed or disallowed model categories.
- Provisional classification is disabled by default. Local experiments require the explicit `TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING=1` switch. The switch does not make the service safe for untrusted input.
- No score threshold is enabled in the service.

## Release blockers

1. **Scope rejection:** Gather a larger, independently reviewed set of both valid and unsupported requests representative of the intended channel. Include banking-adjacent unknowns, urgent wording, and unusual but valid banking queries. Separate calibration from final evaluation before selecting a rule.
2. **Operational tolerance:** The service owner must define acceptable rates of missed unsupported requests and wrongly rejected valid requests, including any stricter rule for fraud or account-access cases. No numerical release gate should be invented after seeing results.
3. **Routing policy:** A responsible owner must confirm real team names, priority semantics, escalation exceptions, and fallback handling. Current values are provisional examples, not an approved queue policy.
4. **Serving controls:** Authentication, traffic limits, privacy-safe logging, retention rules, incident handling, and monitoring have not been implemented or verified. The prior local load test is not a capacity or security assessment.

Until these blockers are addressed, keep the classification switch off outside controlled local testing. If real requests are supplied for evaluation, remove personal data before sharing and retain a separately held-out final set.
