# SSF AI Toolkit (ssfaitk)

A modular, production‑ready Python toolkit for AI in **small‑scale fisheries** (SSF).  
It standardizes common models and utilities so projects can share code and scale quickly.

## Features (initial)
- **Effort classification**: fishing vs. non‑fishing activity within a trip.
- **Gear prediction**: start with 3 gear types (Timor‑Leste); expandable to Zanzibar/Kenya datasets.
- **Vessel type prediction**: motorized vs. non‑motorized.
- Consistent **model interface** (`fit/predict/save/load`) with typed dataclasses.
- **CLI**: `ssfaitk ...` to run inference and utilities from the shell.
- **Evaluation** helpers and model registry.
- **Config** via Pydantic‑style settings (no runtime deps required).

> Package name: `ssfaitk`. You can rename the project anytime.

## Quickstart
```bash
# from repo root
pip install -e .[dev]

# CLI
ssfaitk --help
ssfaitk effort predict --input data/sample_tracks.parquet --output outputs/effort_preds.parquet
ssfaitk gear predict --input data/sample_trips.parquet --output outputs/gear_preds.parquet
ssfaitk vessel predict --input data/sample_trips.parquet --output outputs/vessel_preds.parquet

# Python API
from ssfaitk.models import EffortClassifier, GearPredictor, VesselTypePredictor
clf = EffortClassifier.load_default()
y = clf.predict(df)
```

## Project layout
```
.
├─ src/ssfaitk/                 # library source
│  ├─ models/                   # model families
│  │  ├─ effort/
│  │  ├─ gear/
│  │  └─ vessel/
│  ├─ data/                     # schemas + loaders
│  ├─ utils/                    # io, logging, geospatial helpers
│  ├─ eval/                     # metrics & reports
│  └─ cli.py                    # entrypoint for `ssfaitk`
├─ tests/                       # pytest suites
├─ examples/                    # notebooks & recipes
├─ .github/workflows/           # CI
├─ pyproject.toml               # build, deps, tooling
└─ ...
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). Please read the [Code of Conduct](CODE_OF_CONDUCT.md).

## License
Apache‑2.0 © 2025 WorldFish contributors and the SSF community.
