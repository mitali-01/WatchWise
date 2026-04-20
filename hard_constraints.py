import pandas as pd


class ConstraintEngine:
    def __init__(self, df: pd.DataFrame, users: list):
        self.df = df.copy()
        self.users = users

    def _check_genre_exclude(self, genres, excluded):
        if pd.isna(genres):
            return 0
        genre_list = genres.split("|")
        return int(any(g in genre_list for g in excluded))

    def _check_language(self, lang, allowed):
        if pd.isna(lang):
            return 1
        return int(lang not in allowed)

    def _check_runtime(self, runtime, max_runtime):
        if pd.isna(runtime):
            return 0
        return int(runtime > max_runtime)

    def compute_user_violation(self, df, user):
        violations = pd.Series(0.0, index=df.index)
        constraints = user.get("constraints", {})

        if "genre_exclude" in constraints:
            violations += df["genres"].apply(
                lambda x: self._check_genre_exclude(x, constraints["genre_exclude"])
            )

        if "language_include" in constraints and "original_language" in df.columns:
            violations += df["original_language"].apply(
                lambda x: self._check_language(x, constraints["language_include"])
            )

        if "runtime_max" in constraints and "runtime" in df.columns:
            violations += df["runtime"].apply(
                lambda x: self._check_runtime(x, constraints["runtime_max"])
            )

        return violations

    def apply(self):
        df = self.df.copy()
        violation_cols = []

        for i, user in enumerate(self.users):
            col = f"user_{i}_violation"
            df[col] = self.compute_user_violation(df, user)
            violation_cols.append(col)

        df["total_violation"] = df[violation_cols].sum(axis=1)
        return df