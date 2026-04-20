import pandas as pd
import numpy as np


class ScoringEngine:
    def __init__(self, df: pd.DataFrame, users: list, alpha=0.5):
        self.df = df.copy()
        self.users = users
        self.alpha = alpha
        self.GENRE_MULTIPLIER = 2.5
        self.GENRE_PENALTY = 1.5

    def compute_user_score_with_breakdown(self, df, preferences):
        score = pd.Series(0.0, index=df.index)

        breakdown = {}

        # --- Genre ---
        genre_weights = preferences.get("genre_weights", {})
        genre_match_score = pd.Series(0.0, index=df.index)
        genre_presence = pd.Series(0.0, index=df.index)

        for genre, weight in genre_weights.items():
            col = f"genre_{genre}"
            if col in df.columns:
                genre_presence += df[col]
                genre_match_score += weight * df[col]

        num_genres = df["genres"].apply(
            lambda x: len(x.split("|")) if pd.notna(x) else 1
        )

        genre_score = genre_match_score / num_genres

        if genre_weights:
            no_match_penalty = (genre_presence == 0).astype(float)
            genre_score -= self.GENRE_PENALTY * no_match_penalty

        genre_score = self.GENRE_MULTIPLIER * genre_score
        score += genre_score
        breakdown["genre_score"] = genre_score

        # --- Runtime ---
        runtime_score = pd.Series(0.0, index=df.index)
        if "runtime_weight" in preferences and "runtime_norm" in df.columns:
            runtime_score = preferences["runtime_weight"] * df["runtime_norm"]
            score += runtime_score
        breakdown["runtime_score"] = runtime_score

        # --- Popularity ---
        popularity_score = pd.Series(0.0, index=df.index)
        if "popularity_weight" in preferences and "popularity_norm" in df.columns:
            popularity = df["popularity_norm"].clip(0, 0.8)
            popularity_score = preferences["popularity_weight"] * popularity
            score += popularity_score
        breakdown["popularity_score"] = popularity_score

        # --- Recency ---
        recency_score = pd.Series(0.0, index=df.index)
        if "recency_weight" in preferences and "movie_age_norm" in df.columns:
            recency = (1 - df["movie_age_norm"]).clip(0, 0.8)
            recency_score = preferences["recency_weight"] * recency
            score += recency_score
        breakdown["recency_score"] = recency_score

        return score, breakdown
    
    def apply_user_penalty_with_breakdown(self, score, df, user_idx):
        col = f"user_{user_idx}_violation_scaled"

        if col in df.columns:
            penalty = df[col] ** 2
            return score - self.alpha * penalty, penalty

        return score, pd.Series(0.0, index=df.index)

    def score_users(self):
        user_scores = []
        user_breakdowns = []

        for i, user in enumerate(self.users):
            prefs = user.get("preferences", {})

            score, breakdown = self.compute_user_score_with_breakdown(self.df, prefs)
            score, penalty = self.apply_user_penalty_with_breakdown(score, self.df, i)

            breakdown["penalty"] = penalty
            breakdown["final_user_score"] = score

            user_scores.append(score)
            user_breakdowns.append(breakdown)

        return user_scores, user_breakdowns