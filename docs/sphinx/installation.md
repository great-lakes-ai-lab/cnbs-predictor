# Installation

`cnbs-predictor` is distributed as a conda environment plus a set of Jupyter
notebooks. These steps mirror the project README.

## Prerequisites

- Conda (Anaconda or Miniconda)
- Python 3.11+

On **macOS**, work from a **Terminal** window. On **Windows**, use the
**Anaconda Prompt**. If you don't already have Anaconda or Miniconda, install it
from <https://www.anaconda.com/download>.

## Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/great-lakes-ai-lab/cnbs-predictor.git
   cd cnbs-predictor
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda env create -f requirements/environment.yml
   conda activate nbs_env
   ```

3. **Register the environment as a Jupyter kernel:**

   ```bash
   python -m ipykernel install --user --name nbs_env --display-name "Python (nbs_env)"
   ```

```{note}
`requirements/environment.yml` is the full environment, including the
TensorFlow stack used for *training* new models. A lighter
`requirements/environment-test.yml` (no TensorFlow) is used by the automated
test suite — see {doc}`usage` and the project `CONTRIBUTING.md`.
```

## Building these docs locally

The documentation builds with a small, pip-only set of dependencies (the heavy
compiled libraries are mocked, so they are not required just to build docs):

```bash
pip install -r docs/sphinx/requirements.txt
cd docs/sphinx
make html
```

The rendered site lands in `docs/sphinx/_build/html/index.html`.
