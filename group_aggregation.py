import pandas as pd
import numpy as np


class GroupAggregator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def aggregate(self, user_scores, method="min"):
        """
        user_scores: list of pd.Series
        method: 'average', 'min', 'product'
        """

        scores_matrix = np.vstack(user_scores)

        if method == "average":
            final_scores = scores_matrix.mean(axis=0)

        elif method == "min":
            final_scores = scores_matrix.min(axis=0)

        elif method == "product":
            final_scores = scores_matrix.prod(axis=0)

        else:
            raise ValueError("Invalid aggregation method")

        return pd.Series(final_scores, index=self.df.index)

    def recommend(self, user_scores, top_k=10, method="min"):
        final_scores = self.aggregate(user_scores, method)

        df = self.df.copy()
        df["final_score"] = final_scores

        df = df.sort_values("final_score", ascending=False)

        return df.head(top_k)