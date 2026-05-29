# Wiki drafts

These markdown files are **starter content for the GitHub Wiki**, not part of
the published documentation site. The GitHub Wiki lives in a separate git
repository (`cnbs-predictor.wiki.git`) that can't be updated through a normal
pull request, so the workflow is:

1. Open the repo's **Wiki** tab on GitHub.
2. Create a page whose title matches the filename here (e.g. `Onboarding.md`
   → a page titled **Onboarding**).
3. Paste in the contents, tweak as needed, and save.

## How the Wiki and the docs site divide responsibilities

- **Read the Docs (Sphinx)** = the *reference manual*: installation, usage, and
  the API reference generated from docstrings. It versions with the code and is
  built from `docs/sphinx/`.
- **GitHub Wiki** = the *living knowledge base*: onboarding, design rationale,
  FAQ, and a decision log. Browser-editable, no PR required, good for notes that
  evolve faster than the code.

When something is "how the code works," it belongs in docstrings → Sphinx. When
it's "how we think about the project / why we made a choice / how to get
started," it belongs in the Wiki.

## Pages in this folder

- `Home.md` — Wiki landing page.
- `Onboarding.md` — environment setup, running tests, branch conventions.
- `Design-Notes.md` — development history, dataset decisions, modeling
  rationale, and the smoke-check vs. skill-validation distinction.
- `Experiments.md` — experiment configurations explored during development.
- `Skill-Metrics.md` — skill-validation results (RMSE / R² / bias) across
  experiments and model families.
- `FAQ.md` — stub, grow as questions recur.
- `Decision-Log.md` — stub, append dated decisions.
