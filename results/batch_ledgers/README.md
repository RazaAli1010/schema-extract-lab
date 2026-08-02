# Batch ledgers (F2 provenance)

Retired copies of `data/labeled/_batches.json` — the in-flight ledger the teacher
writes while driving the OpenAI Batch API. They live here rather than beside the
cache because `data/**` is gitignored (see `.gitignore`), and these are the only
record of what was actually submitted and paid for.

Filenames carry the `prompt_sha` the batches were submitted under, which is what
makes them attributable to a specific prompt.

| file | prompt_sha | covers |
|---|---|---|
| `eval_pool-944bd674b0bc38a3.json` | `944bd674b0bc38a3` | the original 361 eval_pool labels, produced by the **pre-fix** prompt |
| `train-dev-c79c6f8c56e78dfd.json` | `c79c6f8c56e78dfd` | the train + dev relabel under the **fixed** prompt (see F2 "Delivered") |

## Reading them

`batches[]` entries are one per submitted batch. `status` is the last value a
live poller wrote, so it is **not** authoritative after a run is killed — a
cancelled or completed batch can be frozen at `validating` here. Ask the API for
current truth; treat these as a record of intent and outcome, not live state.

`harvested: true` means the results were appended to `_teacher_cache.jsonl`.

The train/dev ledger has 32 entries for 5,300 documents because 21 of them
`failed` on the org's 2,000,000 enqueued-token limit and were resubmitted. Those
failures were never processed and never billed; only the 10 `completed` entries
correspond to spend. That ratio is an artifact of `TEACHER_MAX_INFLIGHT_BATCHES`
being 4 while one 500-document batch is ~1.4M input tokens, so only one batch
fits under the ceiling at a time.

`main-0031` is the batch cancelled at 0/500, whose 500 documents were dropped —
it is why `train.jsonl` holds 4,500 rows.
