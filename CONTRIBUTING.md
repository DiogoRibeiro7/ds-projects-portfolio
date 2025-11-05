# Contributing to Data Science Portfolio

Thank you for your interest in contributing to this data science portfolio! This document provides guidelines for contributing to make the process smooth and effective.

## 🎯 Types of Contributions

We welcome several types of contributions:

### 📊 New Case Studies
- **Domain expansion**: Healthcare, finance, marketing, operations
- **New datasets**: Public datasets with interesting problems
- **Industry applications**: Real-world business cases
- **Methodological demos**: New statistical techniques

### 🔧 Code Improvements
- **Bug fixes**: Corrections to existing code
- **Performance**: Optimization of statistical computations
- **Documentation**: Better explanations and examples
- **Testing**: Unit tests for helper functions

### 📚 Educational Content
- **Tutorials**: Step-by-step learning guides
- **References**: Academic papers and resources
- **Translations**: Notebooks in other languages
- **Glossary**: Statistical terms and concepts

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- Jupyter Lab or VS Code with Jupyter extension

### Local Setup
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/data-science-portfolio.git
cd data-science-portfolio

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development tools
pip install black isort mypy pytest

# 6. Create a new branch
git checkout -b feature/your-feature-name
```

## 📝 Contribution Guidelines

### Code Standards

#### Python Code
- **Type hints**: Use type annotations for all functions
- **Docstrings**: Document all functions with clear parameter descriptions
- **Error handling**: Include appropriate error checking
- **Formatting**: Use `black` for code formatting
- **Imports**: Use `isort` for import organization

Example function:
```python
def two_prop_ztest(
    x1: int, 
    n1: int, 
    x2: int, 
    n2: int, 
    two_sided: bool = True
) -> Tuple[float, float]:
    """Two-sample z-test for proportions.
    
    Parameters
    ----------
    x1, n1 : int
        Success count and sample size for group 1.
    x2, n2 : int  
        Success count and sample size for group 2.
    two_sided : bool, default=True
        Whether to perform two-sided test.
        
    Returns
    -------
    z_stat : float
        Z-statistic value.
    p_value : float
        P-value for the test.
        
    Raises
    ------
    ValueError
        If sample sizes are invalid.
    """
    # Implementation here
```

#### Jupyter Notebooks
- **Clear structure**: Use markdown headers for organization
- **Explanations**: Markdown before each code section explaining the approach
- **Interpretations**: Markdown after results explaining what they mean
- **Clean outputs**: Remove unnecessary debug outputs before committing
- **Reproducibility**: Set random seeds for consistent results

### Notebook Structure Template
```markdown
# Project Title — Clear Problem Statement

Brief description of the business problem and approach.

## 1) Setup and Data Loading
- Import statements
- Helper function definitions
- Data loading and basic inspection

## 2) Exploratory Data Analysis
- Data quality checks
- Visualizations
- Summary statistics

## 3) Statistical Analysis
- Hypothesis formulation
- Method selection and justification
- Results and interpretation

## 4) Business Recommendations
- Decision framework
- Risk assessment
- Next steps
```

### Dataset Guidelines

#### Using External Data
- **Public datasets only**: No proprietary or personal data
- **Proper attribution**: Credit data sources
- **Data documentation**: Explain column meanings and context
- **Reasonable size**: Keep datasets under 100MB when possible

#### Simulated Data
- **Realistic parameters**: Based on industry standards
- **Clear generation process**: Document how data was created
- **Reproducible**: Use fixed seeds
- **Well-documented**: Explain what each variable represents

## 🔍 Review Process

### Before Submitting
1. **Run all notebooks**: Ensure they execute without errors
2. **Check formatting**: Run `black` and `isort` on Python files
3. **Verify outputs**: Clear and re-run all cells
4. **Test edge cases**: Verify functions handle boundary conditions
5. **Spell check**: Review markdown for typos

### Pull Request Process
1. **Descriptive title**: Clearly state what the PR does
2. **Detailed description**: Explain the motivation and approach
3. **Link issues**: Reference any related GitHub issues
4. **Testing**: Describe how you tested the changes
5. **Screenshots**: Include plots or outputs if relevant

### PR Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature/case study
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] All notebooks run successfully
- [ ] Code follows style guidelines
- [ ] Added appropriate tests

## Checklist
- [ ] Code is well-documented
- [ ] Commit messages are clear
- [ ] No sensitive data included
```

## 📊 Adding New Case Studies

### Choosing Topics
Good case study topics should:
- **Solve real problems**: Address actual business questions
- **Demonstrate methods**: Show statistical techniques clearly
- **Be educational**: Teach concepts step-by-step
- **Include challenges**: Show how to handle common issues

### Required Elements
1. **Problem statement**: Clear business question
2. **Data description**: Source and variable explanations
3. **Methodology**: Statistical approach with justification
4. **Results**: Findings with proper interpretation
5. **Limitations**: Acknowledge assumptions and constraints
6. **Recommendations**: Actionable business insights

### Statistical Requirements
- **Hypothesis testing**: Proper null/alternative formulation
- **Effect sizes**: Not just p-values, but practical significance
- **Confidence intervals**: Uncertainty quantification
- **Assumptions**: Check and document statistical assumptions
- **Multiple comparisons**: Appropriate corrections when needed

## 🐛 Reporting Issues

### Bug Reports
Please include:
- **Environment**: Python version, OS, package versions
- **Steps to reproduce**: Minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable

### Feature Requests
Please include:
- **Use case**: Why is this needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional context**: Any relevant background

## 📚 Resources

### Statistical Learning
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman
- [Trustworthy Online Controlled Experiments](https://experimentguide.com/) - Kohavi, Tang, Xu
- [Causal Inference: The Mixtape](https://mixtape.scunning.com/) - Scott Cunningham

### Code Style
- [PEP 8](https://pep8.org/) - Python style guide
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

### Jupyter Best Practices
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html)
- [Data Science Project Organization](https://drivendata.github.io/cookiecutter-data-science/)

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Individual notebook acknowledgments
- Release notes for significant contributions

## 📞 Contact

Questions about contributing? Reach out:
- **GitHub Issues**: For technical questions
- **Email**: dfr@esmad.ipp.pt
- **Discussions**: Use GitHub Discussions for general questions

Thank you for helping make this portfolio a valuable resource for the data science community! 🚀
