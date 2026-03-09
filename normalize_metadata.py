import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

INPUT_PATH = DATA_DIR / "movies_master.parquet"
OUTPUT_PATH = DATA_DIR / "movies_features.parquet"


GENRE_MAP = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Horror": "Horror",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western",
    "Documentary": "Documentary",
    "History": "History",
    "Music": "Music",

    "Science Fiction": "SciFi",
    "Sci-Fi": "SciFi",

    "Children": "Family",
    "Family": "Family",

    "Film-Noir": "Crime",
}

TARGET_GENRES = sorted(set(GENRE_MAP.values()))


def normalize_genres(df):

    print("Normalizing genres...")

    def clean_genres(genre_str):

        if pd.isna(genre_str):
            return []

        genres = genre_str.split("|")

        normalized = []

        for g in genres:

            g = g.strip()

            if g in GENRE_MAP:
                normalized.append(GENRE_MAP[g])

        return list(set(normalized))

    df["genres_clean"] = df["genres"].apply(clean_genres)

    for genre in TARGET_GENRES:
        df[f"genre_{genre}"] = df["genres_clean"].apply(lambda g: int(genre in g))

    return df


def encode_language(df):

    print("Encoding language...")

    df["language"] = df["language"].fillna("unknown")

    language_map = {
        lang: idx for idx, lang in enumerate(sorted(df["language"].unique()))
    }

    df["language_id"] = df["language"].map(language_map)

    return df, language_map


def normalize_runtime(df):

    print("Normalizing runtime...")

    mu = df["runtime"].mean()
    sigma = df["runtime"].std()

    df["runtime_norm"] = (df["runtime"] - mu) / sigma

    return df


def normalize_year(df):

    print("Normalizing year...")

    CURRENT_YEAR = 2025

    df["movie_age"] = CURRENT_YEAR - df["year"]

    mu = df["movie_age"].mean()
    sigma = df["movie_age"].std()

    df["movie_age_norm"] = (df["movie_age"] - mu) / sigma

    return df


def normalize_popularity(df):

    print("Normalizing popularity...")

    df["popularity_log"] = np.log1p(df["popularity"])

    mu = df["popularity_log"].mean()
    sigma = df["popularity_log"].std()

    df["popularity_norm"] = (df["popularity_log"] - mu) / sigma

    return df


def finalize_dataset(df):

    print("Finalizing feature dataset...")

    feature_cols = [
        "movieId",
        "tmdbId",
        "title",
        "language",
        "language_id",
        "runtime",
        "runtime_norm",
        "year",
        "movie_age",
        "movie_age_norm",
        "mean_rating",
        "rating_count",
        "popularity",
        "popularity_norm",
    ]

    genre_cols = [f"genre_{g}" for g in TARGET_GENRES]

    df = df[feature_cols + genre_cols]

    return df


def main():

    print("Loading master dataset...")
    df = pd.read_parquet(INPUT_PATH)

    df = normalize_genres(df)

    df, language_map = encode_language(df)

    df = normalize_runtime(df)

    df = normalize_year(df)

    df = normalize_popularity(df)

    df = finalize_dataset(df)

    print("Saving feature dataset...")
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Dataset saved:", OUTPUT_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()