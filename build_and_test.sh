# assumes a python environment has been created and activated
echo "Testing rust" && \
(cd rust/ && cargo test --workspace) && \
# pip install -qe ".[dev]" && \ 
# assumes `pip install -qe ".[dev]"` has been run already
echo "Building python API" && \
maturin develop --release && \
echo "Running python tests" && \
pytest -v tests && \
echo "Verifying that demos run" && \
SHOW_PLOTS=false pytest -v applications/demos/*demo*.py && \ 
echo "Complete success!"