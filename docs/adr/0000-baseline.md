# ADR 0000 — Project Baseline Snapshot

**Date:** 2026-07-21
**Status:** Accepted

## Context

Before beginning the production-hardening transformation of FraudLens, we
capture a baseline snapshot of the project's test health, coverage, and known
issues. This document serves as the starting point against which all future
improvements are measured.

## Baseline Test Results

| Metric | Value |
|--------|-------|
| **Total tests** | 193 |
| **Passed** | 190 |
| **Failed** | 1 |
| **Skipped** | 2 (Autoencoder — TensorFlow not installed) |
| **Coverage** | 74% |
| **Duration** | 73.89s |

## Known Failures (All Resolved)

1. ~~`TSNE.__init__() got an unexpected keyword argument 'n_iter'`~~ → **Fixed.**
   Replaced `n_iter=1000` with `max_iter=1000, learning_rate='auto'` in
   `src/fraudlens/analysis/eda.py`.

2. ~~`Seaborn pairplot KeyError: 'Legitimate'`~~ → **Fixed.**
   `palette` dict keys now correctly derived from `COLORS`/`LABELS` constants
   instead of using mismatched case.

## Remaining Warnings (not blocking)

- Pydantic v2 deprecation: `min_items`/`max_items` should be `min_length`/
  `max_length` (in `api/schemas.py`)
- XGBoost: `use_label_encoder` parameter ignored
- LogisticRegression: `n_jobs` has no effect in sklearn 1.8+
- ConvergenceWarning: lbfgs needs more iterations or scaling

## Audit Scores (from comprehensive audit)

| Category | Score (0–10) |
|----------|:------------:|
| Architecture | 5 |
| Code Quality | 6 |
| Readability | 7 |
| Scalability | 2 |
| Maintainability | 5 |
| Performance | 4 |
| Security | 3 |
| Documentation | 7 |
| Testing | 5 |
| DevOps | 4 |
| UI/UX | 6 |
| Developer Experience | 5 |
| Production Readiness | 2 |
| **OVERALL** | **4.8** |

## Transformation Roadmap

See [execution plan](../README.md) for the phased transformation:

1. **Phase 0:** Baseline, rename, tooling (current)
2. **Phase 1:** Security (auth, rate limiting, secrets)
3. **Phase 2:** Architecture & code quality refactor
4. **Phase 3:** Data layer (Postgres, migrations)
5. **Phase 4:** Performance (async SHAP, caching, load tests)
6. **Phase 5:** ML quality (tuning, model card)
7. **Phase 6:** API design (versioning, RFC 7807)
8. **Phase 7:** Frontend (live demo, fix mock data)
9. **Phase 8:** Testing (85% coverage, integration tests)
10. **Phase 9:** DevOps (multi-stage, CD, K8s)
11. **Phase 10:** Observability (metrics, tracing, dashboards)
12. **Phase 11:** Configuration (pydantic-settings)
13. **Phase 12:** Error handling & resilience
14. **Phase 13:** Documentation & portfolio polish

## Target State

After all phases, the target is:

- All 193+ tests passing, coverage ≥ 85%
- Security: auth, rate limiting, validated inputs, no CVEs
- Architecture: dependency injection, no globals, clean layering
- DevOps: multi-stage Docker under 400MB, CI/CD, K8s manifests
- Documentation: ADRs, model card, system design doc, live demo video
- Overall score: target 9+/10 across all categories
