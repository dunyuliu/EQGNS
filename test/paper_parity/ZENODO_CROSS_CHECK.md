# Zenodo cross-check -- status: BLOCKED: network/size

Ran 2026-09-24 via `python3 test/paper_parity/zenodo_hash_check.py`.

## What worked
- Network access to Zenodo's REST API succeeded:
  `GET https://zenodo.org/api/records/17095311` returned the record manifest
  (15 files) with no error.

## Why it's blocked
1. **Checksum granularity/algorithm mismatch.** Zenodo's manifest exposes
   MD5 checksums of the packed `.zip` archives (`M1.model.rollout.zip`,
   `M2.model.rollout.zip`, `M3.model.rollout.zip`, `M{1,2,3}.train.valid.test.zip*`),
   not sha256 of the individual `model-<step>.pt` / `test.npz` files inside
   them. `baseline_M{1,2,3}.json` records sha256 of the *extracted* files.
   A real diff therefore requires downloading every relevant archive,
   unzipping it, and re-hashing the extracted files with sha256 -- it
   cannot be done from the manifest alone.
2. **Impractical download size at measured throughput.** The archives
   needed to reconstruct M1+M2+M3's checkpoints and test sets total
   **10.33 GB**. A live 20MB range-request sample of `M1.model.rollout.zip`
   measured **389.2 KB/s** sustained throughput from this environment,
   giving an ETA of **~7.4 hours** for the full 10.33 GB -- far beyond what
   is practical for this PR's scope (correctness/completeness over
   runtime, but this is a one-off verification step, not part of the
   gate's steady-state cost).

## What would unblock this
- Re-run `zenodo_hash_check.py` from an environment with faster/cheaper
  egress to zenodo.org (e.g. a cloud VM co-located with Zenodo's CDN, or
  a scheduled overnight job) -- the script's throughput measurement will
  automatically re-evaluate and proceed with the full download+extract+
  sha256 diff if the resulting ETA is judged practical (currently
  hardcoded at a 30-minute budget in the script; adjust
  `if eta_hours > 0.5` if a longer budget is acceptable).
- Alternatively, ask the Zenodo depositor (repo authors) to publish
  sha256 (not just MD5) of the individual `model-*.pt`/`test.npz` files
  directly in the deposit metadata, which would make this check possible
  without downloading the archives at all.

## Not done (explicitly, not silently)
- No partial/sampled hash comparison was substituted for the real check.
  A "BLOCKED" verdict here means exactly that: the cross-check has not
  been performed, not that it passed or is assumed to pass.
