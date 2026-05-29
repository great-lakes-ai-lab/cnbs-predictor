# Onboarding

A practical checklist for getting set up to develop on cnbs-predictor.

## 1. Environment

The project uses a conda environment named `nbs_env`.

```bash
git clone https://github.com/great-lakes-ai-lab/cnbs-predictor.git
cd cnbs-predictor
conda env create -f requirements/environment.yml
conda activate nbs_env
python -m ipykernel install --user --name nbs_env --display-name "Python (nbs_env)"
```

`requirements/environment.yml` is the full environment (includes the TensorFlow
stack used for *training*). There is also a lighter
`requirements/environment-test.yml` — no TensorFlow — used by the automated test
suite and continuous integration.

## 2. Running the tests

```bash
# Fast, offline tests (what CI runs on every PR):
pytest -m "not network"

# The live NOAA/AWS reachability checks (run on a schedule in CI):
pytest -m network
```

Tests that hit live external endpoints are marked `@pytest.mark.network` so they
can be excluded from the default run — a NOAA outage shouldn't block a PR. See
`CONTRIBUTING.md` for the full rationale.

## 3. Branch naming

From `CONTRIBUTING.md`, branches use a type prefix + initials + issue number +
short description:

- `feature/` — e.g. `feature/dj_123_add-forecasting-model`
- `bugfix/`  — e.g. `bugfix/dj_456_fix-csv-import`
- `docs/`    — e.g. `docs/dj_101_update-readme`
- `test/`    — e.g. `test/dj_102_add-unit-tests`

## 4. Continuous integration

On every pull request to `main`, GitHub Actions builds the lightweight test
environment and runs the offline suite. The check grows automatically as new
test files land — no workflow changes needed. The live-network tests run on a
daily schedule instead.

## 5. Documentation

- **Reference docs** (install/usage/API) are built with Sphinx from
  `docs/sphinx/` and published on Read the Docs. To build locally:
  ```bash
  pip install -r docs/sphinx/requirements.txt
  cd docs/sphinx && make html
  ```
- **This Wiki** holds onboarding, design notes, FAQ, and the decision log.
