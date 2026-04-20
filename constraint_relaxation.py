import pandas as pd


class ConstraintRelaxation:
    def __init__(self, df: pd.DataFrame, users: list, lambda_val=0.5):
        self.df = df.copy()
        self.users = users
        self.lambda_val = lambda_val

    def apply_relaxation(self):
        df = self.df.copy()
        scaled_cols = []

        for i, user in enumerate(self.users):
            base_col = f"user_{i}_violation"
            scaled_col = f"user_{i}_violation_scaled"

            if base_col not in df.columns:
                continue

            flexibility = user.get("flexibility", {}).get("constraint_tolerance", 0.3)

            df[scaled_col] = df[base_col] * (1 / (flexibility + 1e-3))
            scaled_cols.append(scaled_col)

        if scaled_cols:
            df["total_violation"] = df[scaled_cols].sum(axis=1)
        else:
            df["total_violation"] = 0

        # stricter filtering to remove junk
        threshold = self.lambda_val * len(self.users) * 0.7
        return df[df["total_violation"] <= threshold]