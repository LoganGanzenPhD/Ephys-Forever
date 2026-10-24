# Ephys-Forever

Open-source, cross-platform analysis of patch-clamp and other electrophysiology
recordings. Development guidance is in [CLAUDE.md](CLAUDE.md) and the test layout
is described in [tests/README.md](tests/README.md).

```
pip install -e ".[dev]"
python scripts/make_synthetic.py   # regenerate synthetic test data
pytest
```
