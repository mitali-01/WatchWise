# WatchWise: Constraint-Aware Group Movie Recommender

## Overview

WatchWise is a constraint-aware group recommendation system designed to suggest movies that satisfy group requirements while maximizing overall satisfaction.

Unlike traditional recommenders, this system explicitly models:

* Hard constraints (must satisfy)
* Relaxable constraints (λ-controlled)
* Soft preferences (scoring-based)

---

## Problem Statement

Group recommendations are challenging due to:

* Conflicting user preferences
* Strict constraints leading to empty results
* Lack of fairness in standard recommendation systems

This project addresses these issues using a hybrid constraint + optimization framework.

---

## System Architecture

Pipeline:

User Input
→ Constraint Filtering
→ λ-based Relaxation
→ Preference Scoring
→ Group Aggregation
→ Top-K Recommendations

---

## Mathematical Formulation

### Constraint Filtering

M₁ = {m ∈ M | satisfies hard constraints}

### Relaxation

V(m) = Σ violationᵢ(m)
M₂ = {m ∈ M₁ | V(m) ≤ λT}

### Scoring

Sᵤ(m) = wᵤᵀ f(m)

### Penalized Score

Sᵤ'(m) = Sᵤ(m) − αV(m)

### Group Aggregation

Score(m) = minᵤ Sᵤ'(m)

---

## Features Used

* Genres (one-hot encoded)
* Runtime (normalized)
* Popularity (normalized)
* Release year (normalized)

---

## Project Structure

```
src/
  constraint_engine/
    hard_constraints.py
    constraint_relaxation.py
    scoring.py
    group_aggregation.py

  pipeline/
    run_recommender.py

data/ (not included)
```

---

## Dataset

* MovieLens 25M
* TMDB Movies Dataset

(Datasets not included due to size)

---

## How to Run

1. Place datasets in `data/`
2. Run preprocessing scripts
3. Execute:

```
python src/pipeline/run_recommender.py
```

---

## Key Contributions

* Constraint-aware recommendation system
* λ-based relaxation for feasibility
* Fairness-aware group aggregation
* Modular and extensible architecture

---

## Future Work (Week 2)

* Learn user preferences from ratings
* Add evaluation metrics
* Improve ranking with embeddings
* Optimize group fairness strategies

---