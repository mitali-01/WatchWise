import pandas as pd

class ConstraintEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split_constraints(self, constraints):
        strict = []
        relaxable = []

        for c in constraints:
            if c["priority"] == 1:
                strict.append(c)
            elif c["priority"] == 2:
                relaxable.append(c)

        return strict, relaxable

    def apply_constraints(self, constraints):
        df = self.df.copy()

        for c in constraints:
            field = c["field"]
            op = c["op"]
            value = c["value"]

            if field not in df.columns:
                continue

            if op == "==":
                df = df[df[field] == value]

            elif op == "!=":
                df = df[df[field] != value]

            elif op == "<=":
                df = df[df[field] <= value]

            elif op == ">=":
                df = df[df[field] >= value]

            elif op == "<":
                df = df[df[field] < value]

            elif op == ">":
                df = df[df[field] > value]

            elif op == "in":
                df = df[df[field].isin(value)]

            elif op == "not_in":
                df = df[~df[field].isin(value)]

            elif op == "genre_in":
                df = df[df["genres"].apply(
                    lambda x: any(g in x.split("|") for g in value)
                    if pd.notna(x) else False
                )]

            elif op == "genre_not_in":
                df = df[df["genres"].apply(
                    lambda x: all(g not in x.split("|") for g in value)
                    if pd.notna(x) else True
                )]

        return df

    def filter(self, constraints):
        strict, relaxable = self.split_constraints(constraints)

        # Apply strict constraints only
        filtered_df = self.apply_constraints(strict)

        return filtered_df, relaxable