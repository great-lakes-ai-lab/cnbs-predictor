# FAQ

Common questions. Add entries as they come up in issues, reviews, or onboarding.

## Do I need TensorFlow to run a forecast?

No. Inference uses scikit-learn models loaded from `.joblib` files. TensorFlow is
only needed for the model *training* workflow. The test environment
(`requirements/environment-test.yml`) omits it entirely.

## The CI "network" tests didn't run on my PR — is that a problem?

No. The live NOAA/AWS tests are marked `network` and are intentionally excluded
from the per-PR run so an upstream outage can't block a merge. They run on a
daily schedule instead (and can be triggered manually once the workflow is on
`main`).

## Where do the docs live — Wiki or Read the Docs?

Both, by design. Install/usage/API reference → Read the Docs (built from the
code). Onboarding, rationale, FAQ, decisions → this Wiki. See [[Home]].

---

> _Add new Q&A below as they arise._
