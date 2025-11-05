# 📊 Data Science Portfolio

> **Professional end-to-end data science projects showcasing real-world analysis, experimentation, and statistical rigor.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

--------------------------------------------------------------------------------

## 🎯 Overview

This repository contains a curated collection of **production-ready** data science projects that demonstrate:

- **Statistical rigor** with proper experimental design and hypothesis testing
- **End-to-end thinking** from problem framing to deployment recommendations
- **Reproducible research** with clear documentation and reusable code
- **Business impact** through actionable insights and decision frameworks

Perfect for **data scientists**, **analysts**, **researchers**, and **students** looking for industry-standard examples of statistical analysis and experimentation.

## 🚀 Featured Projects

### 🧪 A/B Testing & Experimentation Suite

A comprehensive collection of professional A/B testing notebooks covering modern statistical methods:

Project                                                                       | Description                                                | Methods                                                      | Dataset
----------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ | --------------------
**[Professional A/B Playbook](ab_testing/ab_testing_playbook_pro.ipynb)**     | Complete A/B testing framework with advanced methods       | Z-tests, GLM, CUPED, Bootstrap, Bayesian, Sequential Testing | Udacity + Criteo
**[E-commerce Conversion](ab_testing/ab_testing_ecommerce_playbook.ipynb)**   | Landing page optimization with business decision framework | Frequentist + Bayesian, Logistic Regression, Revenue Impact  | Udacity Landing Page
**[Mobile Game Retention](ab_testing/ab_testing_cookie_cats_playbook.ipynb)** | Cookie Cats gate placement analysis                        | Retention Analysis, GLM, CUPED, Power/MDE                    | Cookie Cats
**[Pricing Optimization](ab_testing/ab_testing_pricing_playbook.ipynb)**      | Price elasticity and revenue impact analysis               | Price Elasticity, Revenue Analysis, CUPED, Causal Inference  | Simulated Pricing
**[Modern Multi-Method](ab_testing/ab_testing_modern_multi.ipynb)**           | Multiple case studies with uplift modeling                 | Uplift Analysis, T-learner, Qini Curves                      | Multiple Datasets

#### 🛠️ Advanced Methods Covered

- **Variance Reduction**: CUPED (Controlled-experiment Using Pre-Experiment Data)
- **Sequential Testing**: Pocock, O'Brien-Fleming, Lan-DeMets α-spending
- **Bayesian Methods**: Beta-Binomial models, posterior distributions
- **Multiple Testing**: Holm correction, hierarchical testing
- **Uplift Modeling**: T-learner, DR-learner, Qini evaluation
- **Causal Inference**: IPW, Doubly-Robust estimation
- **Business Decision**: Parametrized decision frameworks

## 📁 Repository Structure

```
.
├── README.md                     # You are here
├── ab_testing/                   # A/B testing & experimentation
│   ├── ab_testing_playbook_pro.ipynb    # 🎯 Comprehensive professional playbook
│   ├── ab_testing_ecommerce_playbook.ipynb
│   ├── ab_testing_cookie_cats_playbook.ipynb
│   ├── ab_testing_pricing_playbook.ipynb
│   └── ab_testing_modern_multi.ipynb
├── data/                         # Sample datasets and data loaders
├── src/                          # Reusable Python modules
├── envs/                         # Environment configurations
└── docs/                         # Additional documentation
```

## 🏃‍♂️ Quick Start

### Option 1: Browse Online

Click on any notebook link above to view it directly on GitHub with full formatting and outputs.

### Option 2: Run Locally

1. **Clone the repository**

  ```bash
  git clone https://github.com/diogoribeiro7/data-science-portfolio.git
  cd data-science-portfolio
  ```

2. **Set up environment**

  ```bash
  # Using conda
  conda create -n ds-portfolio python=3.9
  conda activate ds-portfolio
  pip install -r requirements.txt

  # Or using pip + venv
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

3. **Launch Jupyter**

  ```bash
  jupyter lab
  ```

4. **Start with the main playbook** Open `ab_testing/ab_testing_playbook_pro.ipynb` for a comprehensive overview.

## 📊 Technical Stack

### Core Libraries

- **Statistical Analysis**: `scipy`, `statsmodels`, `numpy`
- **Data Manipulation**: `pandas`, `polars`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`
- **Bayesian Analysis**: Custom implementations with `numpy`

### Key Features

- **Type-safe code** with comprehensive type hints
- **Modular design** with reusable helper functions
- **Professional documentation** with clear explanations
- **Reproducible results** with fixed random seeds
- **Industry standards** following best practices

## 🎓 Learning Path

### For Beginners

1. Start with **[E-commerce Playbook](ab_testing/ab_testing_ecommerce_playbook.ipynb)** - covers fundamentals
2. Move to **[Cookie Cats Analysis](ab_testing/ab_testing_cookie_cats_playbook.ipynb)** - adds retention metrics
3. Explore **[Pricing Optimization](ab_testing/ab_testing_pricing_playbook.ipynb)** - business applications

### For Intermediate Users

1. **[Professional Playbook](ab_testing/ab_testing_playbook_pro.ipynb)** - comprehensive methods
2. **[Modern Multi-Method](ab_testing/ab_testing_modern_multi.ipynb)** - advanced techniques

### For Advanced Practitioners

- Deep dive into variance reduction techniques (CUPED)
- Explore sequential testing and α-spending functions
- Study uplift modeling and causal inference methods
- Implement custom Bayesian models

## 🏆 Why This Portfolio?

### 🔬 **Statistical Rigor**

- Proper experimental design with SRM checks
- Multiple testing corrections
- Power analysis and MDE calculations
- Confidence intervals and effect sizes

### 💼 **Business Focus**

- Decision frameworks with clear recommendations
- Revenue impact analysis
- Risk assessment and business trade-offs
- Executive summary templates

### 🛠️ **Production Ready**

- Clean, documented code with type hints
- Modular helper functions
- Error handling and edge cases
- Reproducible environments

### 📚 **Educational Value**

- Clear explanations before and after code
- Multiple approaches to the same problem
- Common pitfalls and how to avoid them
- Links to academic references

## 🤝 Contributing

Contributions are welcome! Please see <CONTRIBUTING.md> for guidelines.

### Ways to Contribute

- **Add new case studies** with different domains
- **Improve existing notebooks** with additional methods
- **Fix bugs** or enhance documentation
- **Add new statistical methods** or visualizations
- **Translate notebooks** to other languages

## 📖 Additional Resources

### Academic References

- Kohavi, R., Tang, D., & Xu, Y. (2020). _Trustworthy Online Controlled Experiments_
- Deng, A., et al. (2013). _Improving the Sensitivity of Online Controlled Experiments_
- Athey, S., & Imbens, G. (2017). _The Econometrics of Randomized Experiments_

### Online Resources

- [Exp-Platform.com](https://exp-platform.com/) - Microsoft's experimentation blog
- [Netflix Tech Blog](https://netflixtechblog.com/) - A/B testing articles
- [Uber Engineering](https://eng.uber.com/) - Causal inference posts

## 👨‍💼 About the Author

**Diogo Ribeiro** | Lead Data Scientist & Researcher

- 🎓 Master in Mathematics
- 🏫 Teacher & Researcher at ESMAD - Escola Superior de Média Arte e Design
- 💼 Lead Data Scientist at Mysense.ai
- 🔬 ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
- 📧 Contact: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- 💻 GitHub: [@diogoribeiro7](https://github.com/diogoribeiro7)

## 📄 License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## 🌟 Acknowledgments

- **Udacity** for providing excellent public datasets
- **Criteo** for the uplift modeling dataset
- **Open source community** for statistical libraries
- **Academic researchers** for methodological foundations

--------------------------------------------------------------------------------

<div align="center"><strong>⭐ If you find this portfolio helpful, please consider giving it a star! ⭐</strong>

<a href="https://github.com/diogoribeiro7/data-science-portfolio/issues">Report Bug</a> • <a href="https://github.com/diogoribeiro7/data-science-portfolio/issues">Request Feature</a> • <a href="CONTRIBUTING.md">Contribute</a></div>
