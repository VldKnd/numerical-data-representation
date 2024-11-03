# numerical-data-representation
Encodings to represent numerical data

## Dependencies installation

**All the given commands should be run from repository root.**

### uv

I have been using [uv](https://astral.sh/blog/uv) dependecies manager. With it you can just run:
```
uv sync
```
and it should install all of needed dependencies in .venv folder. If you dont use it, read on!

### Venv

I have been installing all the dependencies in virtual environment. It can be created with:
```bash
python -m venv .venv
```
After creating .venv you need to activate it with:
```bash
source .venv/bin/activate
```
### Pip python dependencies
To install dependencies with pip, run:
```bash
pip install .
```
After this all the code can be executed with python from .venv `$REPO_ROOT/.venv/bin/python`. ( You can make sure that terminal is using the correct interpreter by running `which python` ). 

## Repository structure:
```
numerical-data-representation
├── assignment
├── notebooks # Folder with jupyter notebooks. 
|    ├── evaluation.ipynb # Evaluation of logarithmic embeddings on some simple use cases.
|    └── plot_embedding_relations.ipynb # Comparison of different embedding schemes.
├── source
|    ├── numeric_representation/ # Different embedding classes.
|    ├── encode.py # Implementation of needed encode_number function.
|    └── utils.py # Utilities
├── tests
|    └── test_encode_function.py/ # Simple unittests.
├── pyproject.toml # Python dependencies
└── ...
```

## Example of execution:
Minimal working code:
```python
>>> from source.encode import encode_numbers
>>> input = ["1"]
>>>> embeddings = encode_numbers(input=input)
>>> print(embeddings)
tensor([[ 0.8657, -0.4143,  0.8766, -0.3121,  ...  0.8513,  0.5340]])
```

You can find more working examples in `tests/` or `notebooks/`.

To run `tests` you can use
```bash
python -m unittest tests/test_encode_function.py
```