# Full Notebook Test Analysis Report

**Date:** December 30, 2025
**Total Notebooks Tested:** 21
**Pass Rate:** 33.3% (7 passed, 14 failed)
**Total Execution Time:** 111.31 seconds

## Executive Summary

The comprehensive notebook testing revealed significant issues that need to be addressed. While the validation scores are generally good (76-93/100), the execution failures indicate missing dependencies and data files that prevent notebooks from running properly.

## Test Results Overview

### ✅ Passed Notebooks (7)
1. **ab_testing_guardrails_multimetric.ipynb** - Score: 93/100
2. **ab_testing_offpolicy_policy_switch.ipynb** - Score: 88/100
3. **ab_testing_retention_survival.ipynb** - Score: 93/100
4. **ab_testing_sequential_always_valid.ipynb** - Score: 93/100
5. **bandits_exploration_exploitation.ipynb** - Score: 93/100
6. **geo_experiments_playbook.ipynb** - Score: 88/100
7. **meta_analysis_experiment_portfolio.ipynb** - Score: 83/100

### ❌ Failed Notebooks (14)
Grouped by failure type:

#### Missing Data Files (7 notebooks)
- **landing_page data**: ab_testing.ipynb, ab_testing_modern_multi.ipynb, ab_testing_playbook_pro.ipynb
- **cookie_cats.csv**: ab_testing_cookie_cats_playbook.ipynb
- **pricing_ab.csv**: ab_testing_pricing_playbook.ipynb
- **Exited column**: 01_eda_and_baselines.ipynb, bank_churn_clv_retention.ipynb, bank_churn_mlops_api_batch.ipynb, bank_churn_uplift_modelling.ipynb

#### Missing Dependencies (3 notebooks)
- **lightgbm module**: 02_lightgbm_optuna_shap.ipynb, 03_segment_insights.ipynb
- **math.erfcinv function**: ab_testing_modern_playbook.ipynb

#### Syntax Errors (2 notebooks)
- **ab_testing_ecommerce_playbook.ipynb**: Unterminated string literal at line 7
- **experiment_story_templates.ipynb**: Unterminated string literal at line 52

## Key Findings

### 1. Data Availability Issues (67% of failures)
Most failures are due to missing data files. The notebooks expect:
- A/B testing datasets (landing_page, cookie_cats, pricing_ab)
- Bank churn dataset with specific columns (Exited)

### 2. Dependency Issues (21% of failures)
- LightGBM was not installed (now fixed)
- scipy.special.erfcinv might need an import fix

### 3. Code Quality Issues (14% of failures)
- Syntax errors with unterminated strings
- Possible issues with multi-line strings or quotes

### 4. Performance Metrics
- **Average execution time**: 5.28 seconds per notebook
- **Fastest notebook**: experiment_story_templates.ipynb (1.88s)
- **Slowest notebook**: ab_testing_sequential_always_valid.ipynb (19.86s)
- **Memory efficiency**: Very good (-0.03 MB average delta)

## Validation Scores Analysis

All notebooks have good validation scores (76-93/100), indicating:
- Good code structure
- Adequate documentation
- Proper import organization
- Reasonable code quality

The issues are primarily runtime problems, not structural problems.

## Immediate Actions Required

### 1. Fix Data File Issues
```python
# Create data directory structure
mkdir -p ab_testing/data
mkdir -p modern-bank-churn/data

# Add sample data generation or download scripts
# Or provide instructions for obtaining required datasets
```

### 2. Fix Dependencies (Completed)
```bash
✅ pip install lightgbm optuna shap  # Already installed
```

### 3. Fix Syntax Errors
- Review ab_testing_ecommerce_playbook.ipynb line 7
- Review experiment_story_templates.ipynb line 52
- Check for unclosed quotes or f-strings

### 4. Fix Math Function Import
```python
# Add to notebooks using erfcinv:
from scipy.special import erfcinv
# Or use: scipy.special.erfinv (inverse erf function)
```

### 5. Data Schema Issues
The bank churn notebooks expect an 'Exited' column that doesn't exist in the current data file. Either:
- Rename the target column in the data
- Update notebooks to use the correct column name

## Recommendations

### Short-term (Immediate)
1. **Create a data setup script** that downloads or generates all required test data
2. **Fix the two syntax errors** in the notebooks
3. **Update import statements** for math functions
4. **Verify data schema** matches notebook expectations

### Medium-term (This Week)
1. **Add data validation checks** at the beginning of each notebook
2. **Create environment.yml** or requirements.txt specifically for notebooks
3. **Add error handling** for missing data files with helpful messages
4. **Document data requirements** in each notebook's markdown cells

### Long-term (This Month)
1. **Implement data versioning** using DVC or similar
2. **Create notebook templates** with standard imports and setup
3. **Add integration tests** for data pipelines
4. **Set up automated daily/weekly test runs**

## Test Framework Performance

The testing framework itself performed well:
- ✅ Successfully validated all 21 notebooks
- ✅ Captured detailed error messages
- ✅ Generated comprehensive HTML reports
- ✅ Tracked execution metrics accurately
- ✅ Handled failures gracefully

## Next Steps

1. **Fix data availability** - Priority 1
2. **Fix syntax errors** - Priority 1
3. **Re-run tests** after fixes
4. **Set up CI/CD** with the working notebooks
5. **Monitor test trends** over time

## Success Metrics

After fixes, we should aim for:
- **>90% pass rate** for execution tests
- **>85 validation score** for all notebooks
- **<30s average execution time**
- **Zero syntax errors**
- **All data files documented**

---

*Report generated by the Notebook Testing Framework*
*For detailed results, see: test_results/notebook_test_report_20251230_114326.html*