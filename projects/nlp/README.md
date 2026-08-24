# NLP Examples

This folder contains notebook-based NLP examples for text analysis, transformer
workflows, document processing, and utility functions. The material is intended
for portfolio review and local experimentation, not as a packaged NLP platform.

## Contents

- `01_text_analysis_pipeline.ipynb`: text statistics, readability, sentiment,
  topic modeling, classification, entity extraction, summarization, and question
  answering examples.
- `02_advanced_nlp_models.ipynb`: transformer-oriented examples for sentiment,
  named entities, question answering, generation, semantic search, fine-tuning,
  and paraphrasing.
- `03_document_processing.ipynb`: document loading, metadata extraction,
  similarity search, clustering, graph analysis, and visualization examples.
- `utils.py`: shared preprocessing, feature extraction, similarity, evaluation,
  visualization, and data-handling helpers.

## Quick Start

Install the core stack first:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk spacy gensim
python -m spacy download en_core_web_sm
```

Install optional transformer and document-processing dependencies only when
running the notebooks that need them:

```bash
pip install torch transformers datasets evaluate accelerate sentence-transformers
pip install textstat yake-python rake-nltk bertopic top2vec
pip install PyPDF2 python-docx beautifulsoup4 markdown wordcloud plotly networkx
```

Run examples from this folder, or add `projects/nlp` to `PYTHONPATH` before
importing the modules.

```python
from utils import FeatureExtractor, NLPVisualizer, TextPreprocessor

preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess_pipeline(
    text,
    steps=["clean", "tokenize", "stopwords", "lemmatize"],
)

extractor = FeatureExtractor()
features = extractor.extract_basic_features(clean_text)

visualizer = NLPVisualizer()
visualizer.plot_word_cloud(clean_text)
```

## Review Notes

- The notebooks mix classical NLP, transformer examples, and document-analysis
  patterns so reviewers can inspect the breadth of the implementation.
- Optional libraries are heavy; install them selectively for the notebook you
  want to run.
- Treat benchmark-style values inside notebooks as local examples unless the
  notebook states the dataset, model, and evaluation setup.
