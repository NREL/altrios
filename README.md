# ALTRIOS

![Altrios Logo](https://raw.githubusercontent.com/NREL/altrios/main/.github/images/ALTRIOS-logo-web.jpg)

[![Tests](https://github.com/NREL/altrios/actions/workflows/tests.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/tests.yaml) [![wheels](https://github.com/NREL/altrios/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NREL/altrios/actions/workflows/wheels.yaml) ![Release](https://img.shields.io/badge/release-v0.1.0-blue) ![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)

![Model Framework Schematic](https://raw.githubusercontent.com/NREL/altrios/main/.github/images/ALTRIOS_schematic_Alfred_Hicks.png)

The Advanced Locomotive Technology and Rail Infrastructure Optimization System ([ALTRIOS](https://www.nrel.gov/transportation/altrios.html)) is a unique, fully integrated, open-source software tool to evaluate strategies for deploying advanced locomotive technologies and associated infrastructure for cost-effective decarbonization. ALTRIOS simulates freight-demand driven train scheduling, mainline meet-pass planning, locomotive dynamics, train dynamics, energy conversion efficiencies, and energy storage dynamics of line-haul train operations. Because new locomotives represent a significant long-term capital investment and new technologies must be thoroughly demonstrated before deployment, this tool provides guidance on the risk/reward tradeoffs of different technology rollout strategies. An open, integrated simulation tool is invaluable for identifying future research needs and making decisions on technology development, routes, and train selection. ALTRIOS was developed as part of a collaborative effort by a team comprising The National Renewable Energy Laboratory (NREL), University of Illinois Urbana-Champaign (UIUC), Southwest Research Institute (SwRI), and BNSF Railway.

# Installation

## All Users

### Python Setup

1. Python installation options:
   - Option 1 -- Python: https://www.python.org/downloads/. We recommend Python 3.10. Be sure to check the `Add to PATH` option during installation.
   - Option 2 -- Anaconda: we recommend https://docs.conda.io/en/latest/miniconda.html.
1. Setup a python environment. ALTRIOS can work with Python 3.9, or 3.10, but we recommend 3.10 for better performance and user experience. Create a python environment for ALTRIOS with either of two methods:
   - Option 1 -- [Python Venv](https://docs.python.org/3/library/venv.html)
     1. Navigate to the ALTRIOS folder you just cloned or any folder you'd like for using ALTRIOS. Remember the folder you use!
     1. Assuming you have Python 3.10 installed, run `python3.10 -m venv altrios-venv` in your terminal enviroment (we recommend PowerShell in Windows, which comes pre-installed). This tells Python 3.10 to use the `venv` module to create a virtual environment (which will be ignored by git if named `altrios-venv`) in the `ALTRIOS/altrios-venv/`.
     1. Activate the environment you just created to install packages or anytime you're running ALTRIOS:
        - Mac and Linux: `source altrios-venv/bin/activate`
        - Windows: `altrios-venv/Scripts/activate.bat` in a windows command prompt or power shell or `source ./altrios-venv/scripts/activate` in git bash terminal
        - When the environment is activated, your terminal session will have a decorator that looks like `(altrios-venv)`.
   - Option 2 -- Anaconda:
     1. Open an Anaconda prompt (in Windows, we recommend Anaconda Powershell Prompt) and run the command `conda create -n altrios python=3.10` to create an Anaconda environment named `altrios`.
     1. Activate the environment to install packages or anytime you're running ALTRIOS: run `conda activate altrios`.

### ALTRIOS Setup

With your Python environment activated, run `pip install altrios`.

Congratulations, you've completed installation! Whenever you need to use ALTRIOS, be sure to activate your python environment created above.

## Developers

### Cloning the GitHub Repo

Clone the repository:

1. [Download and install git](https://git-scm.com/downloads) -- accept all defaults when installing.
1. Create a parent directory in your preferred location to contain the repo -- e.g. `<USER_HOME>/Documents/altrios_project/`.
1. Open git bash, and inside the directory you created, [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the [ALTRIOS repository](https://github.com/NREL/ALTRIOS) with e.g. `git clone https://github.com/NREL/ALTRIOS.git`.

### Installing the Python Package

Within the ALTRIOS folder, run `pip install -e ".[dev]"`

#### Using Pinned Package Versions

If you want to use pinned package versions to make sure you're environment is the same as the developers, you can do:

```shell
pip install -r requirements-dev.txt
```

#### Updating Pinned Package Versions

If you add a new package as a dependency, you should update the pinned requirements files.
To do this you can install pip tools: `pip install pip-tools` and then:

```shell
pip-compile && pip-compile requirements-dev.in
```

This will generate two files: `requirements.txt` and `requirements-dev.txt` which you can check into the repository.

### Rust Installation

Install Rust: https://www.rust-lang.org/tools/install.

### Automated Building and Testing

There is a shortcut for building and running all tests, assuming you've installed the python package with develop mode. In the root of the `ALTRIOS/` folder, run the `build_and_test.sh` script. In Windows bash (e.g. git bash), run `sh build_and_test.sh`, or in Linux/Unix, run `./build_and_test.sh`. This builds all the Rust code, runs Rust tests, builds the Python-exposed Rust code, and runs the Python tests.

### Manually Building the Python API

Run `maturin develop --release`. Note that not including `--release` will cause a significant computational performance penalty.

### Manually Testing

Whenever updating code, always run `cargo test --release` inside `ALTRIOS/rust/` to ensure that all tests pass. Also, be sure to rebuild the Python API regularly to ensure that it is up to date. Python unit tests run with `python -m unittest discover` in the root folder of the git repository.

### Releasing

To release the package, you can follow these steps:

1. Create a new branch in the format `v<major>.<minor>.<patch>`. For example `v0.2.1`.
1. Update the version number in the `pyproject.toml` file.
1. Open a pull request into the main branch and make sure all checks pass.
1. Once the pull request is merged into the main branch, create a new GitHub release and create a tag that matches the branch name. Once the release is created, a GitHub action will be launched to build the wheels and publish them to PyPI. 

# How to run ALTRIOS

With your activated Python environment with ALTRIOS fully installed, you can run several scripts in `ALTRIOS/applications/demos/`.

You can run the Simulation Manager through a multi-week simulation of train operations with `ALTRIOS/applications/demos/sim_manager_demo.py` by running `python sim_manager_demo.py` in `ALTRIOS/applications/demos/`. This will create a `plots` subfolder in which the plots will be saved. To run interactively, fire up a Python IDE (e.g. [VS Code](https://code.visualstudio.com/Download), [Spyder](https://www.spyder-ide.org/)), and run the file. If you're in VS Code, you can run the file as a virtual jupyter notebook because of the "cells" that are marked with the `# %%` annotation. You can click on line 2, for example, and hit `<Shift> + <Enter>` to run the current cell in an interactive terminal (which will take several seconds to launch) and advance to the next cell. Alternatively, you can hit `<Ctrl> + <Shift> + p` to enable interactive commands and type "run current cell".

# Acknowledgements
 
The ALTRIOS Team would like to thank ARPA-E for financially supporting the research through the LOCOMOTIVES program and Dr. Robert Ledoux for his vision and support. We would also like to thank the ARPA-E team for their support and guidance: Dr. Apoorv Agarwal, Mirjana Marden, Alexis Amos, and Catherine Good.  We would also like to thank BNSF for their cost share financial support, guidance, and deep understanding of the rail industry’s needs.  Additionally, we would like to thank Jinghu Hu for his contributions to the core ALTRIOS code.  We would like to thank Chris Hennessy at SwRI for his support. Thank you to Michael Cleveland for his help with developing and kicking off this project.  
