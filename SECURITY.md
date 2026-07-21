# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅ Active development |
| < 2.0   | ❌ Not supported |

## Reporting a Vulnerability

If you discover a security vulnerability in FraudLens, please:

1. **Do NOT** open a public GitHub issue.
2. Email details to the project maintainers (see GitHub profile).
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

You can expect:
- Acknowledgment within 48 hours
- A timeline for the fix within 5 business days
- Credit in the release notes (if desired)

## Security Audit Checklist

The following items have been addressed in the project's security hardening:

### Authentication & Authorization
- [x] API key authentication via `X-API-Key` header (FastAPI `Depends()`)
- [x] Two key tiers: `admin` and `readonly`
- [x] Keys stored as SHA-256 hashes, never in plaintext
- [x] Admin-only endpoints for key management
- [ ] OAuth2/JWT upgrade path documented in ADR

### Rate Limiting
- [x] `slowapi` middleware on all endpoints
- [x] Stricter limits on `/chat` (LLM cost protection) and `/predict/batch`
- [x] Redis-backed in production; in-memory fallback for dev
- [x] Configurable via `REDIS_URL` environment variable

### Input Validation
- [x] All endpoints use Pydantic models for request bodies
- [x] Field-level validation: `Amount >= 0`, `Time` ranges
- [x] NaN/Inf explicitly rejected with 422
- [x] `min_length`/`max_length` constraints on batch inputs

### Secrets Management
- [x] Secrets loaded from environment variables / `.env` file
- [x] `.env.example` documents all required secrets
- [x] `python-dotenv` for local development
- [ ] AWS Secrets Manager / Docker secrets documented for production

### Transport & Headers
- [x] CORS restricted to explicit origins (no `*`)
- [x] `X-Content-Type-Options: nosniff`
- [x] `X-Frame-Options: DENY`
- [x] `Strict-Transport-Security: max-age=31536000`
- [x] `Cache-Control: no-store` on prediction responses
- [x] `Content-Security-Policy: frame-ancestors 'none'`
- [ ] TLS termination via reverse proxy (Caddy/Nginx) in docker-compose

### Model Security
- [x] SHA-256 checksum verification on model files
- [x] Checksums auto-generated on first load
- [ ] ONNX export for safer serving path (future)

### Infrastructure
- [ ] MLflow behind auth proxy
- [ ] Container image scanning (Trivy/Grype) in CI
- [ ] Secrets never baked into Docker images

## Disclosure Policy

We follow coordinated disclosure: vulnerabilities are reported privately,
fixed in a patch release, and publicly disclosed 30 days after the fix
is released.
