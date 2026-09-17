# Public-release decision record

As of 2026-09-17, the GitHub source repository and the separate Hugging Face adapter repository are public. The owner explicitly accepted that the existing GitHub commit history exposes a personal Gmail address. No history rewrite or force-push was performed. The Hugging Face adapter's commit history was checked separately and uses Hugging Face no-reply and service addresses.

This publication is a research release, not approval for automatic handling of live banking-support requests. Classification is disabled by default; enabling the provisional local service does not resolve the deployment blockers in `production_readiness.md`.

## Remaining release work

1. Choose a source-code license if reuse of the GitHub code is intended. The adapter is separately licensed under Apache 2.0, and the tracked BANKING77 data remains under the publisher's CC BY 4.0 license. Public visibility alone does not license the source code.
2. Keep the GitHub README, the Hugging Face model card, and the actual published files in sync. A fresh clone should point to the public adapter rather than imply that the saved weights are in Git.
3. If the owner later decides to remove the personal email from Git history, plan a separate, explicitly approved rewrite and coordinate any force-push with existing clones. Rewriting reachable commits changes their identifiers and cannot guarantee removal from host caches or prior copies.
4. Do not claim production readiness until representative scope-rejection evaluation, an approved routing policy, and serving controls are completed and reviewed.
