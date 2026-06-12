# PSVCA Phase 0

Phase 0 builds only the measurement infrastructure needed before value certification:
configuration loading, dataset registration, iTransformer-compatible split borders,
pre-test carving, train-fit-only scaling, and run metadata helpers.

Current runnable checks:

```bash
pytest tests/test_no_test_split_access.py -q
python scripts/print_splits.py --config smoke
python scripts/print_splits.py --config ettm1_pl96
python scripts/print_splits.py --config etth1_pl96
python scripts/print_splits.py --config weather_pl96
```

Configs default to the server data root `/home1/lzh/dataset`. For local data, set
`PSVCA_DATA_ROOT`:

```bash
PSVCA_DATA_ROOT=/path/to/dataset python scripts/print_splits.py --config weather_pl96
```

The formal loader path does not read the iTransformer test window. Scaler statistics
are fit only on `train_fit`, then applied to the pre-test values.

Out of scope for Phase 0: linalg kernels, own-base models, surrogates, screening,
certification, FDR, stability certification, model inputs, and any consumer model.
