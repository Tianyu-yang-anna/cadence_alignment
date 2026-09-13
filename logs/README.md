# Databricks job logs (archived)

807 job stdout logs from the CADENCE Databricks runs, gzipped into a single
tarball to keep the repo lean (228MB raw → 27MB compressed).

Extract:

```bash
tar xzf logs/cadence_logs.tar.gz    # -> logs/<job>.log × 807
```

Log names follow `<stage>-<suffix>.log` (e.g. `train-schC.log`,
`bd3-bd312.log`, `benchgen-*.log`). These are historical run logs; the Volume
`status/` heartbeat files were discarded (2026-09-12 cleanup).
