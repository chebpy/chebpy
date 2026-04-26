# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do NOT report security vulnerabilities through public GitHub issues.**

Please use [GitHub Security Advisories](https://github.com/chebpy/chebpy/security/advisories) to report vulnerabilities privately.

### What to Include

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Affected versions
- Suggested fix (optional)

### What to Expect

- **Acknowledgment** within 48 hours
- **Initial assessment** within 7 days
- **Resolution** of critical issues within 30 days

### Release Security
- **OIDC Publishing**: PyPI trusted publishing without stored credentials
- **Signed Commits**: GPG signing supported for releases
- **Tag Protection**: Releases require version tag validation

## Security Best Practices for Users

When using Rhiza templates in your projects:

1. **Keep Updated**: Regularly sync with upstream templates
2. **Review Changes**: Review template sync PRs before merging
3. **Enable Security Features**: Enable CodeQL and Dependabot in your repositories
4. **Use Locked Dependencies**: Always commit `uv.lock` for reproducible builds
5. **Configure Branch Protection**: Require PR reviews and status checks

## Acknowledgments

We thank the security researchers and community members who help keep Rhiza secure.
