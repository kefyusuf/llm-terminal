# Security reports

Use [GitHub private vulnerability reporting](https://github.com/kefyusuf/llm-terminal/security/advisories/new)
for a suspected vulnerability. Private reporting was verified enabled on
2026-10-04. Include the affected candidate/source version, platform, reproduction
steps and impact. Remove tokens, credentials, private model names and local paths
that are not necessary to reproduce the issue. Do not include secrets in public
issues, pull requests or screenshots.

The project currently has development candidates and no declared long-term
security-support window. Do not infer an SLA or stable-release support from a
candidate CI result. Qualify the specific source and distribution hashes before
using a fix; promotion still requires the release gates in
[release readiness](docs/release-readiness.md).

Download-service transport is restricted to loopback. Optional bearer-token
configuration does not make a remote plaintext bind supported. Model artifacts
and upstream declarations remain untrusted inputs; license metadata is not
permission and transfer validation is not model-format or inference approval.
