# Results table refresh measurements

Run `python scripts/benchmark_results_table.py --report table-report.json` in a prepared development environment. The offline harness prevents inherited mount handlers from starting services, provider discovery or hardware polling. It measures synchronous `refresh_table()` calls, then lets Textual settle outside the measured interval. It uses 100 and 1000 fixture rows, widths 90 and 160, and five repetitions by default (maximum 20).

Cases are unchanged results, one alternating download state, and one changing publisher. The report records visible columns, platform/Python, Git source identity, runtime file fingerprints, individual samples, median time and total clear/update calls. Hashes identify measured files but do not attest a clean checkout. This is a headless structural cost measurement, not terminal FPS, input latency, GPU performance or an inference benchmark.

The 2026-10-04 Windows/Python 3.12.14 trial compared the parent `0208eaa` table implementation with the changed-cell guard in the current worktree. The same isolated harness used five repetitions in both runs. At 1000 rows / width 160, unchanged refreshes made 5000 cell writes before and zero after; one-download refreshes made 5000 before and five after. Median one-download refresh was 9.67 ms before and 6.89 ms after. Structural medians were 46.52 ms before and 52.40 ms after: the change does not demonstrate a structural speed improvement. Local reports are retained in `.venv/table-isolated-before.json` and `.venv/table-isolated-after.json`; these are worktree measurements, not release acceptance evidence.

Width 90 hides the download column. The changed-cell guard skips absent cells; no download write or visible download change should be expected there. Structural refresh still rebuilds the table when ordered keys, rendered metadata or layout change. Regression tests retain ordering/metadata checks and verify selection and scroll preservation for download-only changes at both widths. Numerical timings remain observations, not CI thresholds; mutation counts are the stable behavioral assertions.

| Rows / width | Unchanged before / after (ms) | One download before / after (ms) | Structural before / after (ms) |
|---|---:|---:|---:|
| 100 / 90 | 1.01 / 0.91 | 1.61 / 0.92 | 5.08 / 4.39 |
| 100 / 160 | 1.69 / 0.76 | 1.64 / 1.00 | 6.08 / 5.06 |
| 1000 / 90 | 8.18 / 6.54 | 8.44 / 6.30 | 50.74 / 41.11 |
| 1000 / 160 | 9.77 / 9.49 | 9.67 / 6.89 | 46.52 / 52.40 |
