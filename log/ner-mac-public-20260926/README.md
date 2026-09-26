# Audited Apple Silicon measurements

Measured 2026-09-26. Owner: Axon contributors/agents.
See the [public report](../../docs/bert-token-classification-mac-results.md).

- `public-results.json.gz`: ten shared full-test quality results, both complete
  performance runs (30 processes and 18,000 samples each), aggregates, source
  result hashes, execution order, provenance, versions and condition snapshots.
- `all-case-comparison.csv`: all 120 case comparisons, including each run's three
  process p50s, median, min/max and percentage change. Conditions stay separate.
- `audit.json`: independent integrity/statistics check receipt.
- `build_results.py`: offline audit and portable artifact builder. Requires NumPy;
  run without Python `-O`, since verification uses assertions. Does not execute
  benchmark inference or any code from the imported archives.

Reproduce the audit with the two original result archives extracted below
`log/`. The extracted directories are inputs, not outputs:

```bash
python log/ner-mac-public-20260926/build_results.py \
  --previous log/ner-mac-review-20260926/ner-mac-mlxfix-comparison \
  --quiet log/ner-mac-quiet-review-20260926/ner-mac-quiet-comparison \
  --output log/ner-mac-audit-reproduction
```

The builder checks complete checksum-manifest coverage (134 earlier and 72
confirmation files), preservation of every earlier file, shared environment
versions, fixed checkpoint/dataset identifiers, quality span counts and gates,
all raw timing statistics and throughput, sequential deterministic run order,
identical arguments except output paths, and all supplied comparison rows.
It excludes the preserved preliminary attempt and verifies that none of its
performance files were reused as final results. The original initial MLX quality
records are also checked for preservation.

Archive identities:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `ner-mac-mlxfix-results.tar.gz` | 246642 | `c4b1e22345eca95650800bc9e25327f8daf00219bcbc18c547d27e3836b70da2` |
| `ner-mac-quiet-results.tar.gz` | 167889 | `fdf2ef60e7eb38612d649baa9e9fc548fa1f83a6d9104f33d9cb1e7b4f6907ee` |

Original archives are retained locally; they contain machine-local paths and
process identifiers. This public derivative removes result `run_dir` / `output`
paths and snapshot process IDs, parent IDs, elapsed times, and battery identifiers.
All measurement values remain unchanged. The source hashes refer to the
unsanitized input files, not the derived JSON serialization.

The builder needs those original archives for a full integrity replay. Public
readers can independently recompute all published statistics from the included
raw samples and quality counts without them:

```python
import gzip
import json
import statistics

with gzip.open("log/ner-mac-public-20260926/public-results.json.gz", "rt") as stream:
    data = json.load(stream)
runs = data["performance"]["quiet"]["mlx-metal-fp16"]
process_p50s = [
    statistics.median(row["samples_ms"])
    for run in runs
    for row in run["performance"]["shapes"]
    if (row["batch"], row["length"]) == (1, 128)
]
print(statistics.median(process_p50s))  # approximately 2.472 ms
```

Quality validation here recomputes scoring from recorded counts and checks
reported gates; it does not rerun inference or compare logits independently.
Input bundle integrity was checked on the Mac and is included as reported
provenance. First-call timing includes startup/cache/possible compilation costs;
it is not isolated compile latency. The two conditions use three fresh processes
each, not six interchangeable repetitions, and are never pooled. No Mac/MLX
benchmark was executed on the Linux audit host.
