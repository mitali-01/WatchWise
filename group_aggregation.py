import pandas as pd
import numpy as np


class GroupAggregator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def aggregate(self, user_scores, method="hybrid", beta=0.6):
        scores_matrix = np.vstack(user_scores)

        if method == "average":
            final_scores = scores_matrix.mean(axis=0)

        elif method == "min":
            final_scores = scores_matrix.min(axis=0)

        elif method == "product":
            final_scores = scores_matrix.prod(axis=0)

        elif method == "hybrid":
            avg_score = scores_matrix.mean(axis=0)
            min_score = scores_matrix.min(axis=0)
            max_score = scores_matrix.max(axis=0)
            disagreement = scores_matrix.std(axis=0)


            final_scores = (
                0.4 * avg_score +          # overall satisfaction
                0.4 * min_score +          # fairness (least misery)
                0.2 * max_score -          # allow strong individual preference
                beta * disagreement        # penalize conflict
            )

        else:
            raise ValueError("Invalid aggregation method")

        return pd.Series(final_scores, index=self.df.index)

    def recommend(self, user_scores, user_breakdowns, top_k=10, method="hybrid", beta=0.4):
        final_scores = self.aggregate(user_scores, method, beta)

        df = self.df.copy()
        df["final_score"] = final_scores

        top_df = df.sort_values("final_score", ascending=False).head(top_k)

        return top_df, user_breakdowns