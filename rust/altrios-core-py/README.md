# How to update .pyi file
1. Install mypy: `pip install mypy`
2. Get the html documentation: 
```bash
    cd ALTRIOS/rust/altrios-core
    cargo doc --open
```
3. Go to the python folder in Altrios: 
```bash
    cd ALTRIOS/rust/altrios-core-py/python/
```
4. Generate a new pyi file:
```bash
    stubgen altrios_core_py/
```
5. You should see a new file `out/altrios_core_py.pyi` and within it there will be stubs for all the classes and functions in the `altrios_core_py` module

6. Go to the html documentation that was opened and search the classes, update the `out/altrios_core_py.pyi` accordingly.
