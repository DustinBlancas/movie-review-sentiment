# movie-review-sentiment
NLP binary classifier scoring IMDB movie reviews as positive or negative. Uses TF-IDF features with Logistic Regression and LightGBM, plus a NLTK preprocessing pipeline. Achieves test F1 ≈ 0.90 and ROC-AUC ≈ 0.97 on 50K reviews.
# Movie Review Sentiment Classifier

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DustinBlancas/movie-review-sentiment/blob/main/movie_review_sentiment.ipynb)
[![Render via nbviewer](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/DustinBlancas/movie-review-sentiment/blob/main/movie_review_sentiment.ipynb)

> 👉 **Click a badge above to view the rendered notebook with plots, tables, and results.** GitHub's in-browser preview can be unreliable for larger notebooks; the badges always work.

---

NLP binary-classification model that scores IMDB movie reviews as positive or negative. Built for **Film Junky Union**, an indie-film recommendation community that wanted to auto-flag negative reviews for moderation and use review sentiment as input to its recommendation engine.

## Problem

The community had thousands of unlabeled reviews accumulating each week and no automated way to gauge whether they were favorable or critical. A reliable sentiment classifier would let the team prioritize moderation, surface positively-reviewed films, and feed sentiment signals into recommendation ranking.

**Project requirement:** F1 score ≥ **0.85** on the held-out test set.

## Data

- **Source:** IMDB movie review corpus
- **Size:** ~50,000 labeled reviews (~24K train / ~24K test)
- **Label scheme:** ratings 1–4 mapped to *negative*, 7–10 mapped to *positive*; neutral ratings excluded by design

## Methodology

1. **EDA** — examined class balance and review-length distribution; confirmed roughly balanced classes across train and test.
2. **Text preprocessing** — lowercasing, punctuation handling, NLTK lemmatization, and English stop-word removal.
3. **Feature engineering** — TF-IDF vectorization with up to 100,000 unigram and bigram features.
4. **Model comparison** — trained four models on the same TF-IDF matrix:
   - Logistic Regression (Model 1)
   - Logistic Regression with lemmatized text (Model 3)
   - LightGBM gradient-boosting classifier (Model 4)
   - BERT-based pipeline (Model 5) — implemented but skipped at run time due to GPU requirement
5. **Evaluation** — accuracy, F1, average precision score (APS), and ROC-AUC on the held-out test set.

## Results

| Model | Test Accuracy | Test F1 | Test APS | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (Model 1) | 0.90 | 0.90 | 0.96 | 0.97 |
| LightGBM (Model 4) | 0.88 | 0.88 | 0.95 | 0.95 |

**Multiple models exceeded the 0.85 F1 threshold without needing BERT.** Logistic Regression on TF-IDF features turned out to be the strongest performer on this corpus — a useful reminder that linear models with well-engineered text features can match or beat heavier architectures.

Inspecting the highest-weighted features surfaced intuitive sentiment signals: positive top tokens included words like *excellent*, *wonderful*, *brilliant*, and *favorite*; negative top tokens included *worst*, *waste*, *boring*, and *awful* — useful for stakeholder explainability.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `NLTK` · `LightGBM` · `matplotlib`

## Repository

- [`movie_review_sentiment.ipynb`](./movie_review_sentiment.ipynb) — full notebook: EDA, preprocessing, TF-IDF + model training, evaluation, and feature inspection
