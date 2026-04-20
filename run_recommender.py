import pandas as pd

from hard_constraints import ConstraintEngine
from constraint_relaxation import ConstraintRelaxation
from scoring import ScoringEngine
from group_aggregation import GroupAggregator


def run_recommender(df, users, lambda_val=0.5, alpha=0.5, top_k=10):
    # Step 1 — Compute per-user constraint violations
    constraint_engine = ConstraintEngine(df, users)
    constrained_df = constraint_engine.apply()

    # Step 2 — Apply relaxation (scale violations via flexibility)
    relaxer = ConstraintRelaxation(constrained_df, users, lambda_val=lambda_val)
    feasible_df = relaxer.apply_relaxation()

    # Step 3 — Scoring
    scorer = ScoringEngine(feasible_df, users, alpha=alpha)
    user_scores, user_breakdowns = scorer.score_users()

    # Step 4 — Aggregation
    aggregator = GroupAggregator(feasible_df)
    recommendations, user_breakdowns = aggregator.recommend(
        user_scores,
        user_breakdowns,
        top_k=top_k,
        method="hybrid",
        beta=0.4
    )

    top_idx = recommendations.index
    print("\n=== DEBUG OUTPUT ===\n")

    for idx in top_idx:
        row = recommendations.loc[idx]
        print(f"\nTitle: {row['title']}")
        print(f"Genres: {row['genres']}")

        for u_idx, breakdown in enumerate(user_breakdowns):
            print(f"\n  User {u_idx}:")
            print(f"    Genre: {breakdown['genre_score'].loc[idx]:.3f}")
            print(f"    Runtime: {breakdown['runtime_score'].loc[idx]:.3f}")
            print(f"    Popularity: {breakdown['popularity_score'].loc[idx]:.3f}")
            print(f"    Recency: {breakdown['recency_score'].loc[idx]:.3f}")
            print(f"    Penalty: {breakdown['penalty'].loc[idx]:.3f}")
            print(f"    Final User Score: {breakdown['final_user_score'].loc[idx]:.3f}")

        print(f"\n  Final Group Score: {row['final_score']:.3f}")
        print("-" * 50)
        
    return recommendations


if __name__ == "__main__":
    df = pd.read_parquet("data/movies_features.parquet")

    users = [
        {
            "constraints": {
                "genre_exclude": ["Horror"],
                "language_include": ["English"],
                "runtime_max": 120
            },
            "preferences": {
                "genre_weights": {"Action": 1.0, "Comedy": 0.5},
                "runtime_weight": -0.3,
                "popularity_weight": 0.4,
                "recency_weight": 0.6
            },
            "flexibility": {
                "constraint_tolerance": 0.2
            }
        },
        {
            "constraints": {
                "genre_exclude": ["Horror"],
                "language_include": ["English", "Hindi"],
                "runtime_max": 150
            },
            "preferences": {
                "genre_weights": {"Drama": 1.0, "Romance": 0.5},
                "runtime_weight": -0.2,
                "popularity_weight": 0.3,
                "recency_weight": 0.5
            },
            "flexibility": {
                "constraint_tolerance": 0.5
            }
        }
    ]

    recs = run_recommender(df, users)
    print(recs[["title", "final_score"]])