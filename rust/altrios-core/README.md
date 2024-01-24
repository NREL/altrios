# altrios-core

![Altrios Logo](https://raw.githubusercontent.com/NREL/altrios/main/.github/images/ALTRIOS-logo-web.jpg)

[![Tests](https://github.com/NREL/altrios/actions/workflows/tests.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/tests.yaml) [![wheels](https://github.com/NREL/altrios/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/wheels.yaml?event=release) ![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue) [![Documentation](https://img.shields.io/badge/documentation-custom-blue.svg)](https://nrel.github.io/altrios/) [![GitHub](https://img.shields.io/badge/GitHub-altrios-blue.svg)](https://github.com/NREL/altrios)

This crate is primarily intended to be used as a backend for the [ALTRIOS PyPI package](https://pypi.org/project/altrios/), but it can also function as a dependency for other crates.  

## Developers
If you've updated `altrios-proc-macros`, be sure to publish that crate first and then update the Cargo.toml dependency for this crate.  

To release this crate, you need to be setup as developer for this crate in crates.io.  After making changes and updating the version number in Cargo.toml, follow these steps: 
1. increment the version number in [Cargo.toml](./Cargo.toml)
1. run `git tag ac<major>.<minor>.<patch>`, where `ac<major>.<minor>.<patch>` should look like
   `ac0.1.4`, reflecting whatever the current version is
1. run `cargo publish --dry-run` to make sure everything checks
1. run `cargo publish` to release the update.  

In the future, we may incorporate this into GitHub Actions.  