import pandas as pd
import numpy as np


class ScoringEngine:
    def __init__(self, df: pd.DataFrame, alpha=0.5):
        self.df = df.copy()
        self.alpha = alpha

    def compute_user_score(self, df, preferences):
        score = pd.Series(0, index=df.index)

        # Genre weights (one-hot encoded)
        genre_weights = preferences.get("genre_weights", {})
        for genre, weight in genre_weights.items():
            col = f"genre_{genre}"
            if col in df.columns:
                score += weight * df[col]

        # Numeric preferences (safe checks added)

        if "runtime_weight" in preferences and "runtime_norm" in df.columns:
            score += preferences["runtime_weight"] * df["runtime_norm"]

        if "popularity_weight" in preferences and "popularity_norm" in df.columns:
            score += preferences["popularity_weight"] * df["popularity_norm"]

        # Recency → use movie_age_norm (invert to favor newer movies)
        if "recency_weight" in preferences and "movie_age_norm" in df.columns:
            score += preferences["recency_weight"] * (-df["movie_age_norm"])

        return score

    def apply_penalty(self, score, df):
        if "total_violation" in df.columns:
            return score - self.alpha * df["total_violation"]
        return score

    def score_users(self, users):
        user_scores = []

        for user in users:
            prefs = user["preferences"]

            score = self.compute_user_score(self.df, prefs)
            score = self.apply_penalty(score, self.df)

            user_scores.append(score)

        return user_scores