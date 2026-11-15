# Reproducible validation continuation

`btp-validation` is a separate notebook; all earlier notebooks and source datasets remain unchanged. It runs a fresh bounded sequence with the same evaluation criteria: six M0 repeatability jobs, eight placement jobs, six cache jobs and twelve confirmation jobs. The selected placement is explicitly `M2-R`, and the candidate cache mode is `never`. Dependent stages stop if evidence fails; publishing is not successful evaluation.

The previous offload/cache archive was not saved. Its supplied table remains preliminary, and a new run cannot retrospectively explain the later rejection. Do not mix those observations with fresh records.

## Restart and selection

Each stage appends its exact directory to `stage-paths.json`. The cache stage calls `python -m btp.workflow` with explicit baseline and screening paths. The command recomputes eligibility from raw predictions and provenance, ignores cached approval JSON and writes a new decision document with evidence hashes. An ineligible choice prints the recorded reasons. It never falls back silently or increases the output-variation threshold.

After a kernel restart, restore the runtime, weights and dataset, then set `OUTPUT`, `BASELINE` and `SCREEN` to explicit paths from the preserved session. Reexecute stage helper definitions. Do not pick a report by newest modification time. If the source or runtime changed, the gate must reject combining those runs. Download the session ZIP before ending the session.

## Pending research stages

Fresh GPU results, expanded development/native-size evaluation, held-out locking, persistent model reuse and compact-student training are not completed by this notebook. Asynchronous transfer implementation requires profiling evidence of useful overlap. Compression requires a fixed training protocol and approval for long training. These are outstanding tasks in the implementation plan, not demonstrated improvements.
