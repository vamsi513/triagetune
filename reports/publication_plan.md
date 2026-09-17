# Private-to-public publication plan

Current state: the repository is **private**. The owner chose to keep it private while planning a commit-history rewrite. This plan does not authorize or perform that rewrite or a visibility change.

1. Confirm the exact no-reply commit address to use for past and future commits. Set local commit identity for future work separately; that does not alter old commits.
2. Decide whether to add a source-code license. The tracked BANKING77 data remains under its publisher's CC BY 4.0 license regardless of the software choice. Public visibility can be chosen without a source-code license, but that should be stated plainly.
3. Before rewriting, record the current branch tip and make a recoverable, private backup. Check for collaborators, forks, open requests, tags, and other branches that might still reference old commits.
4. With explicit approval for the history rewrite, perform it in an isolated clone, replacing author and committer email metadata in every reachable project commit while preserving the intended names, dates, messages, and file content. Rewritten commits receive new identifiers and may invalidate old signatures or links.
5. Run all tests, recheck tracked-file and history scans, and verify that no reachable commit retains the personal address. Review the full comparison before any force-push. Coordinate with anyone who has cloned the old history.
6. With explicit approval for the force-push, update the private remote branch, verify the new remote history, and understand that host caches or prior clones may retain old commit objects for some time; rewriting is not a guaranteed erasure mechanism.
7. Only then, with approval to publish the reviewed state, change visibility to public and verify access without authentication. Recheck README, data attribution, license choice, default-off classification, and repository settings from the public view.

The model is still not approved for automatic handling of untrusted requests. Making the source repository public is separate from making the service production-ready.
