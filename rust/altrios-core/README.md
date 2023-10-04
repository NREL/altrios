# altrios-core

This crate is primarily intended to be used as a backend for the [ALTRIOS PyPI package](https://pypi.org/project/altrios/), but it can also function as a dependency for other crates.  

## Developers
To release this crate, you need to be setup as developer for this crate in crates.io.  After making changes and updating the version number in Cargo.toml, you can run `cargo publish --dry-run` to make sure everything checks and then run `cargo publish` to release the update.  

In the future, we may incorporate this into GitHub Actions.  