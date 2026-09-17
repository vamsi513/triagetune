# Engineering audit

Date: 2026-09-17. Scope: the current `main` branch, its reachable commit history, tracked data and reports, local saved artifacts, the service interface, and release documentation. This is an engineering review, not a security certification or deployment approval.

Publication update (2026-09-17): The owner subsequently authorized publication with the existing personal email in Git commit metadata. The GitHub repository and Hugging Face adapter are now public. No history rewrite was performed. The privacy disposition and publication status below describe the original audit decision; the current decision is recorded in `publication_plan.md`.

## Assessment

**Research repository: 7/10. Deployment readiness: 2/10.** These are qualitative engineering judgments, not measured model scores. The repository has a reserved test split, four baselines or model variants, confidence intervals, error analysis, real local serving checks, and explicit negative results. It is not ready for public exposure of live requests because unknown-request rejection is unreliable, routing policy is provisional, and operational controls are missing.

## Checks performed

| Check | Result |
| --- | --- |
| Automated tests | 29 passed locally |
| Python syntax | Source and tests compiled |
| Runtime dependencies | Key local imports succeeded; installed packages reported no broken requirements |
| Source data hashes | Both tracked source files matched the documented expected SHA-256 values |
| Model test fingerprint | Processed test file matched the saved evaluation fingerprint |
| Saved adapter test results | 3,080 rows aligned with test text and labels; 2,479 correct, zero invalid JSON, eight disallowed labels |
| Classical test results | 3,080 rows; 2,608 correct, matching the saved report |
| External scope report | 600 selected examples; frozen-score accept/reject totals add to 600 |
| Tracked-file review | No model weights, local cache, experiment store, or credential file tracked |
| Credential-pattern scan | No common credential or private-key pattern found in tracked work or reachable commit contents; this is not proof that all sensitive content is absent |
| Commit metadata at original audit | A personal email was present in all ten then-reachable commits; the owner later accepted public exposure without a rewrite |

The live model and container were not rerun for this audit. Earlier measured runs remain in the serving reports. The newer default-off classification switch is covered by automated endpoint tests, not a new container run.

## Findings and disposition

| Severity | Finding | Disposition |
| --- | --- | --- |
| Privacy disclosure | Public visibility exposes a personal address in commit metadata. | The owner accepted this disclosure and made the repository public without rewriting history. See `reports/publication_plan.md`. |
| High | Unsupported requests are often forced into banking categories. One frozen score rule accepted 201 of 600 external challenge requests; the same rule would wrongly reject 85 of 3,080 valid banking test requests. | Provisional classification remains off by default. No rejection threshold was enabled. Independently reviewed, representative scope data and an owner-defined error tolerance are needed. |
| High | Priority and team are project-created guesses, not dataset targets or an approved operating policy. | Every route is marked provisional; automatic production routing remains out of scope pending owner review. |
| Medium | No source-code license has been selected. Public visibility alone would not grant reuse rights. | No license was invented. The owner must choose one before claiming this as an open-source software release. Dataset licenses and attribution are separated in `DATA_PROVENANCE.md`. |
| Medium | The saved adapter and classical artifact are not in version control; a fresh clone cannot run the trained service immediately. | README and provenance documentation state this explicitly. Reproduction requires local training or a separately approved artifact publication. |
| Medium | The data audit flagged 422 near-duplicate cross-split pairs at the chosen similarity threshold. | Results remain reported as measured; no test rows were changed after evaluation. These flags may make performance optimistic and need case review before stronger claims. |
| Medium | The service has no authentication, traffic policy, privacy-safe request logging, or incident monitoring. | No public deployment was performed. The default-off switch reduces accidental exposure but is not a substitute for those controls. |
| Low | Serving dependencies use version ranges and the container base uses a mutable tag. | Reproduction is partly protected by model/data hashes and recorded versions, but a release lockfile and fresh container verification are still needed. |

## Corrections made during this review

- Added explicit data-source licenses, attribution, checksums, and redistribution boundaries.
- Added automated report-integrity checks that catch altered source data or stale headline test metrics.
- Excluded common environment and private-key files from future commits and container build contexts.
- Kept the repository private during the original audit while the owner considered commit metadata exposure. The owner later approved publication with that disclosure.

No measured accuracy, rejection rate, or runtime value was changed to improve the rating. The release blockers above remain visible rather than being hidden by a passing test suite.
