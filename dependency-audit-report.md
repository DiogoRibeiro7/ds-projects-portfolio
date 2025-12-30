# Dependency Audit Report

**Date:** December 29, 2025
**Project:** ds-projects-portfolio
**Python Environment:** Python 3.13 with venv

## Executive Summary

The dependency audit revealed significant issues that require immediate attention:
- **158 outdated packages** found
- **14-19 security vulnerabilities** detected across multiple packages
- **1 dependency conflict** identified
- **3 local packages** could not be audited

## Critical Findings

### 1. Security Vulnerabilities (High Priority)

#### Critical Vulnerabilities Found:
- **mlflow 3.3.1**: Deserialization vulnerability allowing arbitrary code execution (GHSA-wf7f-8fxf-xfxc)
- **nbconvert 7.16.6**: Windows-specific vulnerability allowing unauthorized code execution (GHSA-xm59-rqc7-hhvf)
- **urllib3 2.5.0**: Two compression-related vulnerabilities:
  - Unbounded decompression chain (GHSA-gm62-xv2j-4w53) - Fixed in 2.6.0
  - Excessive resource consumption (GHSA-2xpw-w6gg-jr37) - Fixed in 2.6.0
- **werkzeug 3.1.3**: Windows path traversal vulnerability (GHSA-hgf8-39gv-g3f2) - Fixed in 3.1.4

#### Additional Vulnerabilities (from safety scan):
- **starlette 0.47.3**: 1 vulnerability
- **sqlparse 0.5.3**: 1 vulnerability
- **pymongo 4.6.0**: 1 vulnerability
- **peewee 3.18.2**: 1 vulnerability

### 2. Dependency Conflicts

- **scikit-survival 0.24.1** requires `osqp<1.0.0,>=0.6.3` but `osqp 1.0.4` is installed
  - This could cause runtime errors or unexpected behavior

### 3. Outdated Packages (Sample of Critical Updates)

Major version updates available for key packages:
- **Faker**: 19.12.0 → 39.1.0 (major version jump)
- **fastapi**: 0.116.1 → 0.128.0
- **huggingface-hub**: 0.35.3 → 1.2.3
- **numpy**: 2.3.2 → 2.4.0
- **pandas**: 2.3.1 → 2.3.3
- **torch**: 2.7.1 → 2.9.1
- **transformers**: 4.54.1 → 4.57.3

### 4. Local Packages Not Audited

The following local packages could not be audited for vulnerabilities:
- `efficient-plackett-luce (0.1.0)`
- `loss-functions-gof (1.0.0)`
- `psod (0.1.0)`

## Recommendations

### Immediate Actions (Security Critical)

1. **Update vulnerable packages immediately:**
   ```bash
   pip install --upgrade urllib3>=2.6.0 werkzeug>=3.1.4 nbconvert mlflow
   ```

2. **Fix the dependency conflict:**
   ```bash
   pip install "osqp<1.0.0,>=0.6.3" --force-reinstall
   ```

### Short-term Actions (Within 1 week)

3. **Update all packages with security vulnerabilities:**
   ```bash
   pip install --upgrade starlette sqlparse pymongo peewee
   ```

4. **Update critical outdated packages:**
   ```bash
   pip install --upgrade numpy pandas torch transformers huggingface-hub
   ```

### Long-term Actions

5. **Implement dependency management best practices:**
   - Pin exact versions in requirements.txt for production
   - Use requirements-dev.txt for development dependencies
   - Set up automated dependency updates with Dependabot or Renovate
   - Implement regular security scanning in CI/CD pipeline

6. **Consider dependency updates strategy:**
   - Test major version updates in isolated environments
   - Review breaking changes for major updates (especially Faker 19→39)
   - Create a testing plan for validating updates

## Automated Update Commands

### Update all packages at once (use with caution - test thoroughly):
```bash
pip list --outdated | grep -v "Package" | grep -v "^-" | awk '{print $1}' | xargs -n1 pip install --upgrade
```

### Update only packages with known fixes:
```bash
pip install --upgrade urllib3>=2.6.0 werkzeug>=3.1.4 nbconvert mlflow starlette sqlparse pymongo peewee
```

### Generate updated requirements file:
```bash
pip freeze > requirements-updated.txt
```

## Testing Recommendations

After updates:
1. Run all unit tests: `pytest`
2. Verify notebook functionality
3. Test data science pipelines
4. Check for breaking changes in ML workflows
5. Validate API endpoints if using FastAPI

## Monitoring

Consider implementing:
- **safety** or **pip-audit** in pre-commit hooks
- GitHub Security Alerts or Dependabot
- Regular monthly dependency audits
- Automated vulnerability scanning in CI/CD

## Summary Statistics

- Total packages scanned: 341
- Outdated packages: 158 (46%)
- Packages with vulnerabilities: 9
- Total vulnerabilities: 14-19
- Dependency conflicts: 1

## Next Steps

1. **Prioritize security updates** for packages with known vulnerabilities
2. **Test updates** in a development environment
3. **Document any breaking changes** encountered
4. **Update both requirements.txt files** after testing
5. **Set up automated dependency monitoring** for future protection

---

*Report generated on 2025-12-29. For questions or concerns, please contact the development team.*