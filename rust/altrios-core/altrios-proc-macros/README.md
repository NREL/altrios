# altrios-proc-macros

![Altrios Logo](https://raw.githubusercontent.com/NREL/altrios/main/.github/images/ALTRIOS-logo-web.jpg)

[![Tests](https://github.com/NREL/altrios/actions/workflows/tests.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/tests.yaml) [![wheels](https://github.com/NREL/altrios/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/wheels.yaml?event=release) ![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue) [![Documentation](https://img.shields.io/badge/documentation-custom-blue.svg)](https://nrel.github.io/altrios/) [![GitHub](https://img.shields.io/badge/GitHub-altrios-blue.svg)](https://github.com/NREL/altrios)

This crate contains procedural macros used in [altrios-core](https://crates.io/crates/altrios-core).

## Developers
To release this crate, you need to be setup as developer for this crate in crates.io.  After making changes and updating the version number in Cargo.toml, you can run `cargo publish --dry-run` to make sure everything checks and then run `cargo publish` to release the update.  

In the future, we may incorporate this into GitHub Actions.  