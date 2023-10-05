# assumes a python environment has been created and activated
(cd rust/altrios-core/ && cargo test) && \
# pip install -qe ".[dev]" && \ 
# assumes `pip install -qe ".[dev]"` has been run already
maturin develop --release && \
pytest -v tests && \
(cd applications/demos/ && \
echo "Running sim_manager_demo.py" && \
python sim_manager_demo.py && \
echo "Running rollout_demo.py" && \
python rollout_demo.py) && \
echo "Everything worked!"