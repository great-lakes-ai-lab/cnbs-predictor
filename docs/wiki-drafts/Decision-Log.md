# Decision Log

A running record of notable project decisions: what was decided, when, and why.
Append new entries at the top. Keep each entry short — link to an issue/PR for
detail.

---

### 2026-05 — CI runs the test suite; no environment-build check

**Decision:** Run the pytest suite automatically on PRs (lightweight conda env,
no TensorFlow). Do **not** resurrect the old environment-build GitHub Action.

**Why:** The old check tried to certify the users' exact Windows runtime — an
unanswerable question that produced red we learned to ignore. Running the tests
asks an actionable question instead ("does the code pass on a clean checkout?").
Live-network tests are split behind a `network` marker so NOAA outages don't
block PRs.

---

### 2026-05 — Sphinx + Read the Docs for reference; Wiki for knowledge base

**Decision:** API/reference docs via Sphinx on Read the Docs (mocking heavy
compiled deps so the build is pip-only). Onboarding/rationale/FAQ in the GitHub
Wiki.

**Why:** Clean separation — reference docs version with the code; the knowledge
base evolves independently and is browser-editable. Mocking compiled deps
(cfgrib/eccodes, netCDF4, ...) avoids a slow, fragile conda build on RTD.

---

> _Template for new entries:_
>
> ```
> ### YYYY-MM — short title
> **Decision:** what was decided.
> **Why:** the reasoning / alternatives rejected.
> _(link to issue/PR)_
> ```
