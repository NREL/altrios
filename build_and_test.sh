(cd rust/ && cargo test --workspace --exclude=uom) && \
pip install -qe ".[dev]" && \
pytest -v tests