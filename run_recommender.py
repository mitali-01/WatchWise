import pandas as pd

from hard_constraints import ConstraintEngine
from constraint_relaxation import ConstraintRelaxation
from scoring import ScoringEngine
from group_aggregation import GroupAggregator


def run_recommender(df, users, constraints, lambda_val=0.5, alpha=0.5, top_k=10):
    # Step 1 — Hard constraints
    constraint_engine = ConstraintEngine(df)
    filtered_df, relaxable_constraints = constraint_engine.filter(constraints)

    # Step 2 — Relaxation
    relaxer = ConstraintRelaxation(
        filtered_df,
        relaxable_constraints,
        lambda_val=lambda_val
    )
    feasible_df = relaxer.apply_relaxation()

    # Step 3 — Scoring
    scorer = ScoringEngine(feasible_df, alpha=alpha)
    user_scores = scorer.score_users(users)

    # Step 4 — Aggregation
    aggregator = GroupAggregator(feasible_df)
    recommendations = aggregator.recommend(
        user_scores,
        top_k=top_k,
        method="min"
    )

    return recommendations


if __name__ == "__main__":
    df = pd.read_parquet("data/movies_features.parquet")

    constraints = [
        {"field": "genres", "op": "genre_not_in", "value": ["Horror"], "priority": 1},
        {"field": "runtime", "op": "<=", "value": 120, "priority": 2, "flexibility": 30},
    ]

    users = [
        {
            "preferences": {
                "genre_weights": {"Action": 1.0, "Comedy": 0.5},
                "runtime_weight": -0.3,
                "popularity_weight": 0.4,
                "recency_weight": 0.6
            }
        },
        {
            "preferences": {
                "genre_weights": {"Drama": 1.0, "Romance": 0.5},
                "runtime_weight": -0.2,
                "popularity_weight": 0.3,
                "recency_weight": 0.5
            }
        }
    ]

    recs = run_recommender(df, users, constraints)

    print(recs[["title", "final_score"]])