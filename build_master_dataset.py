import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

MOVIES_PATH = DATA_DIR / "movies.csv"
LINKS_PATH = DATA_DIR / "links.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TMDB_PATH = DATA_DIR / "TMDB_movie_dataset_v11.csv"

OUTPUT_PATH = DATA_DIR / "movies_master.parquet"


def load_datasets():
    print("Loading datasets...")

    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    tmdb = pd.read_csv(TMDB_PATH, low_memory=False)

    return movies, links, ratings, tmdb


def clean_tmdb(tmdb):
    print("Cleaning TMDB dataset...")

    # Keep only required columns
    tmdb = tmdb[
        [
            "id",
            "title",
            "release_date",
            "runtime",
            "original_language",
            "popularity",
            "genres",
        ]
    ]

    # Rename columns
    tmdb = tmdb.rename(
        columns={
            "id": "tmdbId",
            "original_language": "language",
        }
    )

    # Remove invalid rows
    tmdb = tmdb[tmdb["runtime"] > 0]
    tmdb = tmdb[tmdb["release_date"].notna()]

    # Extract year
    tmdb["year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year

    tmdb = tmdb.drop(columns=["release_date"])

    return tmdb


def aggregate_ratings(ratings):
    print("Aggregating ratings...")

    rating_stats = (
        ratings.groupby("movieId")
        .agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    return rating_stats


def merge_datasets(movies, links, tmdb, rating_stats):
    print("Merging datasets...")

    # MovieLens movies + links
    df = movies.merge(links, on="movieId", how="left")

    # Join ratings
    df = df.merge(rating_stats, on="movieId", how="left")

    # Join TMDB
    df = df.merge(tmdb, on="tmdbId", how="left")

    return df


def filter_movies(df):
    print("Filtering movies...")

    df = df[df["tmdbId"].notna()]
    df = df[df["runtime"].notna()]
    df = df[df["year"].notna()]

    # Remove very obscure movies
    df = df[df["rating_count"] >= 20]

    return df


def finalize_columns(df):
    print("Finalizing dataset...")

    # Prefer TMDB title if available
    df["title"] = df["title_y"].combine_first(df["title_x"])

    def merge_genres(row):
        genres_x = row["genres_x"]
        genres_y = row["genres_y"]

        set_x = set()
        set_y = set()

        if pd.notna(genres_x):
            set_x = {g.strip() for g in genres_x.split("|")}

        if pd.notna(genres_y):
            set_y = {g.strip() for g in genres_y.split(",")}

        merged = set_x.union(set_y)

        if len(merged) == 0:
            return None

        return "|".join(sorted(merged))

    df["genres"] = df.apply(merge_genres, axis=1)

    df = df[
        [
            "movieId",
            "tmdbId",
            "title",
            "genres",
            "runtime",
            "language",
            "year",
            "mean_rating",
            "rating_count",
            "popularity",
        ]
    ]

    return df


def save_dataset(df):
    print("Saving dataset...")

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print("Final size:", len(df))


def main():
    movies, links, ratings, tmdb = load_datasets()

    tmdb = clean_tmdb(tmdb)

    rating_stats = aggregate_ratings(ratings)

    df = merge_datasets(movies, links, tmdb, rating_stats)

    df = filter_movies(df)
    print(df.columns)


    df = finalize_columns(df)

    save_dataset(df)


if __name__ == "__main__":
    main()

# import pandas as pd

# df = pd.read_parquet("data/movies_master.parquet")

# print(df.head(5))