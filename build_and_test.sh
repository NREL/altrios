# assumes a python environment has been created and activated
echo "Testing rust" && \
(cd rust/ && cargo test --workspace) && \
# pip install -qe ".[dev]" && \ 
# assumes `pip install -qe ".[dev]"` has been run already
echo "Building python API" && \
maturin develop --release && \
echo "Running python tests" && \
pytest -v python/altrios/tests && \
echo "Verifying that demos run" && \
pytest -v python/altrios/demos && \
echo "Complete success!"
