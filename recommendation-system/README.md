# MovieLens Recommender – Production-Style System Design

## 1. Overview

This project implements a **multi-model recommender stack** inspired by a production system, using the MovieLens `ml-latest-small` dataset.

Core ideas:

* Use **multiple base recommenders** (collaborative + content).
* Turn all of them into **features**.
* Add simple **user and item statistics**.
* Train a **learning-to-rank meta model** to produce final recommendation scores.

The same pattern scales to larger data, more features, and more complex infra.

---

## 2. Objectives

* Predict which movies a user is likely to **engage with positively**.
* Optimise for **top-K recommendation quality**, not just rating RMSE.
* Combine:

  * Collaborative filtering signals.
  * Content-based signals (genres).
  * Popularity and user-activity signals.
* Provide a structure that maps to a **realistic production architecture**:

  * Offline training.
  * Feature computation.
  * Online scoring.
  * Monitoring and experimentation.

---

## 3. Data

### 3.1 Sources

From `ml-latest-small`:

* `ratings.csv`

  * `userId`, `movieId`, `rating`, `timestamp`
* `movies.csv`

  * `movieId`, `title`, `genres`

(Optionally extend later with `tags.csv` and others.)

### 3.2 Train / Meta / Test Split

To support stacking and LTR:

1. `ratings_df` → `train_full` (80%) + `test` (20%).
2. `train_full` → `base_train` (80%) + `meta_train` (20%).

Roles:

* **base_train**
  Train all base recommenders.

* **meta_train**
  Generate base scores and features; train meta models.

* **test**
  Final offline evaluation of all models and stacks.

This mimics production practice where meta/ranking models are trained on **out-of-sample** predictions from base models.

---

## 4. Base Models Layer

All base models share the same interface:

* `fit(df)` – train on ratings.
* `predict_df(df)` – output scores for `(userId, movieId)` pairs.

### 4.1 BiasModel

Formula:

[\hat r_{ui} = \mu + b_u + b_i]

* `μ` – global average rating.
* `b_u` – user bias.
* `b_i` – item bias.

Role:

* Very stable baseline.
* Captures global effects and systematic tendencies.
* Often acts as a **prior** or stabiliser for other scores.

### 4.2 ItemKNNModel (Item-based CF)

* Construct user–item rating matrix on `base_train`.
* Compute **item–item cosine similarity**.
* For `(u, i)`:

  * Look at items user `u` has rated.
  * Select top-k most similar to item `i`.
  * Score is similarity-weighted average of those ratings, with fallbacks.

Role:

* Captures **local neighbourhood structure** (“people who liked X also liked Y”).
* Complementary to global models like MF.

### 4.3 MatrixFactorizationModel (MF)

Formula:

[\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i]

* Low-rank latent factors for users and items.
* Trained by SGD on explicit ratings.

Role:

* Strong global collaborative signal.
* Typically the **strongest single model** on MovieLens-style data.

### 4.4 GenreContentModel (Content-based)

* Parse `genres` from `movies.csv`.
* Build a **multi-hot genre matrix** per movie.
* For each user:

  * Build a **user profile**: average of genre vectors for movies they rated ≥ 4.0.
  * L2-normalise profile.
* For `(u, i)`:

  * Score = cosine similarity between user profile and item genre vector.

Role:

* Uses **item content** instead of only interactions.
* Helps for:

  * Cold items.
  * Users with clear genre tastes.
* Adds diversity and interpretability.

---

## 5. Meta-Features Layer

For each `(user, movie)` pair, we construct a feature vector combining:

### 5.1 Base Model Scores

From `base_train` and applied on `meta_train` and `test`:

* `score_bias` – BiasModel prediction.
* `score_itemknn` – ItemKNNModel prediction.
* `score_mf` – MF prediction.
* `score_genre` – GenreContentModel similarity.

These are the **main collaborative + content signals**.

### 5.2 User Statistics (from train_full)

Computed on `train_full`:

* `user_n_ratings` – count of ratings by the user.
* `user_mean_rating` – average rating given by the user.

Interpretation:

* Activity level.
* User generosity/harshness.

### 5.3 Item Statistics (from train_full)

* `item_n_ratings` – number of ratings for the movie (popularity).
* `item_mean_rating` – average rating for the movie (perceived quality).

These are simple proxies for **popularity and consensus quality**, which often improve ranking.

All of the above together form the feature matrix used by the meta models.

---

## 6. Meta Models (Ranking Layer)

### 6.1 Target Definition

We move from rating prediction to **binary relevance**:

* Label `y = 1` if `rating ≥ 4.0`.
* Label `y = 0` otherwise.

This aligns with **top-K recommendation** objectives.

### 6.2 Logistic Regression Stack (Baseline)

* Input: base scores + user/item statistics.
* Output: `P(relevant | features)` via logistic regression.

Role:

* Simple, interpretable baseline.
* Learns global weights for each feature.
* Already a valid stacked ranking model.

### 6.3 Gradient Boosting Stack (Learning-to-Rank Style)

* Model: `GradientBoostingClassifier` (tree-based GBDT).
* Input: same feature set as logistic.
* Output: `P(relevant | features)`.

Why:

* Captures **non-linear interactions** between signals.
* Deals well with mixed scales (counts, means, scores).
* Closer to typical production choices (XGBoost, LightGBM).

Example interactions it can learn:

* “If item is very popular, small MF score drop doesn’t matter.”
* “If user is cold (few ratings), rely more on genre + biases than MF.”
* “If genre similarity is high, boost under-rated niche items.”

Final ranking for a given user is obtained by sorting items by this predicted probability.

---

## 7. Offline Evaluation

We evaluate:

* Base models:

  * `BiasModel`
  * `ItemKNNModel`
  * `MFModel`
  * `GenreContent`
* Simple ensemble:

  * `SimpleAverage4` (mean of four base scores)
* Meta models:

  * `LogisticStack`
  * `GBStack` (gradient boosting stack)

Metrics on **test**:

* **Hit-rate@K** – proportion of users with at least one relevant item in top-K.
* **Precision@K** – on average, how many of the top-K are relevant.
* **Recall@K** – on average, what fraction of relevant items per user appear in top-K.

RMSE is still computed for reference, but ranking metrics drive design decisions.

We also inspect:

* **Feature importances** from the GBDT model:

  * To understand which signals drive ranking.
  * To explain behaviour to stakeholders.

---

## 8. Serving Architecture (Conceptual)

This repo is offline only, but maps cleanly to a serving architecture.

### 8.1 Offline Layer (Batch)

* Periodically (e.g. hourly/daily):

  1. Ingest new interactions (ratings).
  2. Recompute user and item statistics.
  3. Retrain or incrementally update:

     * BiasModel.
     * ItemKNNModel (or precompute similarities).
     * MFModel.
     * GenreContent user profiles.
  4. Recompute meta-training data and retrain the meta LTR model if needed.

* Persist:

  * Base model parameters.
  * Genre matrices and user profiles.
  * GBDT model (and logistic model if used as fallback).

### 8.2 Online Serving (Real-time scoring)

Given a user `u` and a candidate set of movies:

1. **Candidate generation**
   Use fast methods, e.g.:

   * ItemKNN top-N from items user liked.
   * Global popular movies.
   * Simple content-based matches by genre.

2. **Feature computation** per candidate:

   * Base scores:

     * `score_bias`, `score_itemknn`, `score_mf`, `score_genre`.
   * User stats:

     * `user_n_ratings`, `user_mean_rating`.
   * Item stats:

     * `item_n_ratings`, `item_mean_rating`.

3. **Ranking**

   * Input features into **GBStack**.
   * Sort by predicted `P(relevant)`.

4. **Response**

   * Return top-K items to the client.

In a real system, base scores and statistics would live in a **feature store** or a cache to avoid heavy computation on each request.

---

## 9. Monitoring and Experimentation

### 9.1 Monitoring

In production you would track:

* Serving metrics:

  * Latency for feature fetch and scoring.
  * Error rates.
* Engagement metrics:

  * CTR on recommended items.
  * Long-term metrics: watch time, conversions, etc.

### 9.2 A/B Testing

To deploy safely:

* Run online experiments comparing:

  * Baseline (e.g. MF-only or SimpleAverage).
  * LogisticStack.
  * GBStack.

Measure:

* Uplift in engagement.
* Impact on long-tail items.
* Fairness across user segments.

---

## 10. Future Extensions

Some natural next steps that fit into this design:

1. **Richer content features**

   * Use `tags.csv`.
   * Build TF-IDF or embedding-based item vectors.
   * Add those scores/vectors as additional features for the meta-model.

2. **More base models**

   * Add LightFM, neural MF, or sequence models.
   * Treat their outputs as new `score_*` columns.

3. **Pairwise / listwise LTR**

   * Replace pointwise GBDT with LambdaMART / LambdaRank using a dedicated LTR library.

4. **Contextual features**

   * Time of day, day of week.
   * Device type.
   * Campaign or entry point.

5. **User segmentation**

   * Build separate meta models for user segments (e.g. new users vs power users) or include segment indicators as features.

---

This system design ties the notebooks together into a coherent, production-shaped story: multiple recommenders feeding a feature layer, then a learning-to-rank meta-model that optimises directly for top-K quality.
