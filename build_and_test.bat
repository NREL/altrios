call conda activate ./envs
cd rust
cargo test --release
cd ..
maturin develop --release
pytest -v tests