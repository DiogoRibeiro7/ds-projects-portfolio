import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def make_interactions(
    n_users: int = 200,
    n_items: int = 120,
    min_items: int = 5,
    max_items: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for user_id in range(n_users):
        n_interactions = rng.integers(min_items, max_items + 1)
        items = rng.choice(n_items, size=n_interactions, replace=False)
        for item_id in items:
            rows.append((user_id, item_id, 1))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "implicit_feedback"])


def train_item_similarity(interactions: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    user_item = interactions.pivot_table(
        index="user_id",
        columns="item_id",
        values="implicit_feedback",
        fill_value=0,
    )
    similarity = cosine_similarity(user_item.T)
    return user_item, similarity


def recommend_items(
    user_item: pd.DataFrame,
    similarity: np.ndarray,
    user_id: int,
    k: int = 5,
) -> list[int]:
    user_vector = user_item.loc[user_id].values
    scores = similarity.dot(user_vector)
    scores = scores - (user_vector * 1e9)  # filter seen items
    top_items = np.argsort(scores)[::-1][:k]
    return top_items.tolist()


def train_test_split_interactions(interactions: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    test_rows = []
    train_rows = []
    for user_id, group in interactions.groupby("user_id"):
        group = group.sample(frac=1.0, random_state=random_state)
        if len(group) < 2:
            train_rows.append(group)
            continue
        test_idx = rng.choice(group.index, size=1, replace=False)
        test_rows.append(group.loc[test_idx])
        train_rows.append(group.drop(test_idx))
    return pd.concat(train_rows), pd.concat(test_rows)


def evaluate_hit_rate(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    k: int = 5,
) -> float:
    user_item, similarity = train_item_similarity(train_interactions)
    hits = 0
    total = 0
    for _, row in test_interactions.iterrows():
        user_id = row["user_id"]
        true_item = row["item_id"]
        if user_id not in user_item.index:
            continue
        recs = recommend_items(user_item, similarity, user_id, k=k)
        hits += int(true_item in recs)
        total += 1
    return hits / total if total else 0.0


def main() -> None:
    interactions = make_interactions()
    train_interactions, test_interactions = train_test_split_interactions(interactions)

    hit_rate = evaluate_hit_rate(train_interactions, test_interactions, k=5)

    user_item, similarity = train_item_similarity(train_interactions)
    example_user = int(user_item.index[0])
    recs = recommend_items(user_item, similarity, example_user, k=5)

    print(f"Hit Rate@5: {hit_rate:.3f}")
    print(f"Example recommendations for user {example_user}: {recs}")


if __name__ == "__main__":
    main()
