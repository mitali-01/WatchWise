import pandas as pd
import numpy as np


class ConstraintRelaxation:
    def __init__(self, df: pd.DataFrame, relaxable_constraints, lambda_val=0.5):
        self.df = df.copy()
        self.constraints = relaxable_constraints
        self.lambda_val = lambda_val

    def compute_violation(self, df, constraint):
        field = constraint["field"]
        op = constraint["op"]
        value = constraint["value"]
        flexibility = constraint.get("flexibility", 1)

        if field not in df.columns:
            return pd.Series(0, index=df.index)

        # Numeric constraints
        if op == "<=":
            violation = np.maximum(0, df[field] - value)

        elif op == ">=":
            violation = np.maximum(0, value - df[field])

        elif op == "<":
            violation = np.maximum(0, df[field] - value)

        elif op == ">":
            violation = np.maximum(0, value - df[field])

        # Categorical constraints
        elif op == "in":
            violation = ~df[field].isin(value)
            violation = violation.astype(int)

        elif op == "not_in":
            violation = df[field].isin(value)
            violation = violation.astype(int)

        elif op == "genre_in":
            violation = df["genres"].apply(
                lambda x: 0 if any(g in x.split("|") for g in value) else 1
                if pd.notna(x) else 1
            )

        elif op == "genre_not_in":
            violation = df["genres"].apply(
                lambda x: 1 if any(g in x.split("|") for g in value) else 0
                if pd.notna(x) else 0
            )

        else:
            violation = pd.Series(0, index=df.index)

        # Normalize numeric violations
        if violation.dtype != int:
            violation = violation / (flexibility + 1e-6)

        return violation

    def apply_relaxation(self):
        df = self.df.copy()

        total_violation = pd.Series(0, index=df.index)

        for c in self.constraints:
            v = self.compute_violation(df, c)
            total_violation += v

        df["total_violation"] = total_violation

        # Threshold
        threshold = self.lambda_val * len(self.constraints)

        feasible_df = df[df["total_violation"] <= threshold]

        return feasible_df