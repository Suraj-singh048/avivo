# Technology FAQ — Avivo Platform

## Q: What programming languages does the Avivo platform support?
A: The Avivo platform supports Python (3.10+), JavaScript/TypeScript (Node.js 20+), and Go 1.21+. REST and gRPC APIs are available for all supported languages.

## Q: How do I reset my API key?
A: Navigate to Settings → API Keys → Revoke and then click Generate New Key. Note: revoking a key immediately invalidates it, so update your applications before revoking.

## Q: What are the API rate limits?
A: Free tier: 100 requests/minute. Pro tier: 1,000 requests/minute. Enterprise: custom limits. Rate limit headers (`X-RateLimit-Remaining`) are included in every response.

## Q: Does Avivo support webhooks?
A: Yes. Webhooks can be configured in the dashboard under Integrations → Webhooks. Events supported include `user.created`, `payment.success`, `payment.failed`, and `subscription.cancelled`. Payloads are signed with HMAC-SHA256.

## Q: How is uptime guaranteed?
A: Avivo maintains a 99.9% SLA for the Pro and Enterprise tiers. Status and incident history are available at status.avivo.com. Free tier users do not receive an SLA guarantee.

## Q: What authentication methods are supported?
A: API Key (header `Authorization: Bearer <key>`), OAuth 2.0 (PKCE flow for web apps), and JWT tokens with RS256 signing. SSO via SAML 2.0 is available for Enterprise accounts.
