# Deep Learning Notebooks

This folder collects exploratory deep-learning notebooks and helper utilities
used as portfolio examples. The material is intended for review and local
experimentation, not as a packaged framework or deployed service.

## Contents

- `01_neural_architecture_search.ipynb`: neural architecture search and
  hyperparameter tuning patterns.
- `02_model_interpretation.ipynb`: model interpretation workflows such as SHAP,
  LIME, and gradient-based explanations.
- `02_transfer_learning_suite.ipynb`: transfer-learning experiments and
  comparison utilities.
- `03_transfer_learning.ipynb`: fine-tuning and adaptation examples.
- `04_experiment_tracking.ipynb`: experiment tracking and reproducibility
  patterns.
- `05_complete_deep_learning_example.ipynb`: end-to-end notebook tying together
  training, evaluation, and reporting.
- `utils.py`: shared helpers for model construction, training loops,
  reproducibility, metrics, and visualization.

## Local Use

Install the repository dependencies first, then add any notebook-specific deep
learning packages required by the notebook you plan to run:

```bash
pip install -r requirements.txt
```

Open notebooks from the repository root so relative imports and data paths work
consistently:

```bash
jupyter lab projects/deep_learning
```

The notebooks may require optional packages such as PyTorch, Optuna, SHAP, LIME,
MLflow, or experiment-tracking clients. Keep those dependencies isolated in a
local environment when reproducing a specific workflow.

## Review Notes

Use this folder to evaluate:

- experiment design and reproducibility practices;
- model selection and comparison workflow;
- interpretability and diagnostics;
- training-loop organization in `utils.py`;
- communication of model behavior through notebook narrative and plots.

For the shortest pass, review `01_neural_architecture_search.ipynb`,
`02_model_interpretation.ipynb`, and `utils.py`.
