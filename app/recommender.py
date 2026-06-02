import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RAW_DATA_PATH = "data/raw/amazon.csv"
CATALOG_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "about_product",
    "rating",
    "discounted_price",
    "img_link",
]


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_series(series):
    if series is None or series.empty:
        return pd.Series(dtype=float)

    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()
    if max_val <= min_val:
        if max_val == 0:
            return pd.Series(0.0, index=series.index)
        return series / (abs(max_val) + 1e-9)
    return (series - min_val) / (max_val - min_val)


def _split_multi_value(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_catalog_and_interactions():
    raw = pd.read_csv(RAW_DATA_PATH)
    raw = raw.dropna(subset=["user_id", "product_id", "product_name", "category", "about_product", "rating"])
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw = raw.dropna(subset=["rating"])
    raw["product_id"] = raw["product_id"].astype(str).str.strip()

    catalog = raw.loc[:, CATALOG_COLUMNS].drop_duplicates(subset=["product_id"], keep="first")
    catalog = catalog.reset_index(drop=True)

    interactions = raw[["user_id", "product_id", "rating"]].copy()
    interactions["user_id"] = interactions["user_id"].apply(_split_multi_value)
    interactions = interactions.explode("user_id")
    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()
    interactions["product_id"] = interactions["product_id"].astype(str).str.strip()
    interactions = interactions[interactions["user_id"].ne("")]
    interactions = interactions.drop_duplicates(subset=["user_id", "product_id"], keep="first")
    interactions = interactions.reset_index(drop=True)

    return catalog, interactions


df, interaction_df = _load_catalog_and_interactions()
product_id_to_idx = pd.Series(df.index, index=df["product_id"]).to_dict()


# =========================
# CONTENT-BASED FILTERING
# =========================

df["brand"] = df["product_name"].astype(str).str.extract(r"^\s*([A-Za-z0-9]+)", expand=False).fillna("")

for col in ["product_name", "category", "about_product", "brand"]:
    df[f"clean_{col}"] = df[col].astype(str).apply(clean_text)

df["content"] = (
    (df["clean_product_name"] + " ") * 3
    + (df["clean_category"] + " ") * 3
    + (df["clean_brand"] + " ") * 2
    + df["clean_about_product"]
)

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=1,
    max_features=12000,
    sublinear_tf=True,
    strip_accents="unicode",
)
tfidf_matrix = tfidf.fit_transform(df["content"])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# =========================
# COLLABORATIVE FILTERING
# =========================

user_item = interaction_df.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    aggfunc="mean",
)

user_means = user_item.mean(axis=1)
user_item_centered = user_item.sub(user_means, axis=0).fillna(0)
user_sim = cosine_similarity(user_item_centered)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

item_means = user_item.mean(axis=0)
item_user_centered = user_item.sub(item_means, axis=1).fillna(0).T
item_sim = cosine_similarity(item_user_centered)
item_sim_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)

product_stats = interaction_df.groupby("product_id")["rating"].agg(["mean", "count"])
product_popularity = (
    0.75 * normalize_series(product_stats["mean"])
    + 0.25 * normalize_series(product_stats["count"])
).sort_values(ascending=False)


def _coerce_history(user_id, user_history=None):
    if user_history is not None:
        if isinstance(user_history, pd.Series):
            history = user_history.dropna().astype(float)
        elif isinstance(user_history, dict):
            history = pd.Series(user_history, dtype=float).dropna()
        else:
            history = pd.Series(dtype=float)
        return history[history > 0]

    if user_id not in user_item.index:
        return pd.Series(dtype=float)

    history = user_item.loc[user_id].dropna().astype(float)
    return history[history > 0]


def _drop_excluded(scores, exclude_items=None):
    if scores.empty:
        return scores
    if exclude_items is None:
        return scores
    return scores.drop(labels=list(exclude_items), errors="ignore")


def _get_popular_items(exclude_items=None):
    scores = _drop_excluded(product_popularity.copy(), exclude_items)
    return scores.sort_values(ascending=False)


def _item_knn_scores(rated_items, k_neighbors=80, min_similarity=0.0):
    scores = pd.Series(dtype=float)
    total_weight = pd.Series(dtype=float)

    for item_id, rating in rated_items.items():
        if item_id not in item_sim_df.index:
            continue

        sims = item_sim_df.loc[item_id].drop(labels=[item_id], errors="ignore")
        sims = sims[sims > min_similarity].nlargest(k_neighbors)
        if sims.empty:
            continue

        rating_weight = max(float(rating) - 3.0, 0.25)
        scores = scores.add(sims * rating_weight, fill_value=0)
        total_weight = total_weight.add(sims.abs(), fill_value=0)

    if scores.empty:
        return scores

    scores = scores / total_weight.replace(0, np.nan)
    return scores.dropna()


def _user_knn_scores(user_id, rated_items, k_neighbors=60, min_similarity=0.0):
    if user_id not in user_sim_df.index:
        return pd.Series(dtype=float)

    sims = user_sim_df.loc[user_id].drop(labels=[user_id], errors="ignore")
    sims = sims[sims > min_similarity].nlargest(k_neighbors)
    if sims.empty:
        return pd.Series(dtype=float)

    neighbor_ratings = user_item.loc[sims.index].sub(user_means.loc[sims.index], axis=0).fillna(0)
    weighted = neighbor_ratings.mul(sims, axis=0).sum(axis=0)
    weights = neighbor_ratings.ne(0).mul(sims.abs(), axis=0).sum(axis=0).replace(0, np.nan)
    scores = (weighted / weights).dropna()
    scores = scores + rated_items.mean()
    return scores[scores > 0]


def _profile_content_scores(rated_items, top_n=300):
    scores = pd.Series(dtype=float)
    for item_id, rating in rated_items.items():
        if item_id not in product_id_to_idx:
            continue
        idx = product_id_to_idx[item_id]
        sims = pd.Series(content_sim[idx], index=df["product_id"])
        sims = sims.drop(labels=[item_id], errors="ignore")
        scores = scores.add(sims * (float(rating) / 5.0), fill_value=0)

    if scores.empty:
        return scores
    return (scores / max(len(rated_items), 1)).sort_values(ascending=False).head(top_n)


def get_cf_scores(
    user_id,
    top_k=15,
    min_similarity=0.0,
    exclude_items=None,
    user_history=None,
    include_fallback=True,
):
    """Collaborative scores using item-item KNN, user-user KNN, and sparse-data fallbacks."""
    user_id = str(user_id).strip()
    rated_items = _coerce_history(user_id, user_history=user_history)

    if rated_items.empty:
        return _get_popular_items(exclude_items) if include_fallback else pd.Series(dtype=float)

    if exclude_items is None:
        exclude_items = set(rated_items.index)
    else:
        exclude_items = set(exclude_items)

    item_scores = _item_knn_scores(rated_items, k_neighbors=max(top_k * 6, 60), min_similarity=min_similarity)
    user_scores = _user_knn_scores(user_id, rated_items, k_neighbors=max(top_k * 4, 40), min_similarity=min_similarity)

    candidate_index = item_scores.index.union(user_scores.index)
    if candidate_index.empty:
        scores = pd.Series(dtype=float)
    else:
        item_norm = normalize_series(item_scores.reindex(candidate_index).fillna(0))
        user_norm = normalize_series(user_scores.reindex(candidate_index).fillna(0))
        scores = 0.75 * item_norm + 0.25 * user_norm

    scores = _drop_excluded(scores, exclude_items)
    scores = scores[scores > 0].sort_values(ascending=False)

    if include_fallback and len(scores) < top_k:
        popularity = normalize_series(_get_popular_items(exclude_items))
        scores = scores.combine_first(popularity)

    return scores.sort_values(ascending=False)


# =========================
# CONTENT-BASED FILTERING
# =========================

def get_cb_scores(product_id, top_n=None, exclude_items=None):
    product_id = str(product_id).strip()
    if product_id not in product_id_to_idx:
        return pd.Series(dtype=float)

    idx = product_id_to_idx[product_id]
    sim_scores = pd.Series(content_sim[idx], index=df["product_id"])
    sim_scores = sim_scores.drop(labels=[product_id], errors="ignore")
    sim_scores = _drop_excluded(sim_scores, exclude_items)
    sim_scores = sim_scores[sim_scores > 0]

    if sim_scores.index.duplicated().any():
        sim_scores = sim_scores.groupby(level=0).max()

    sim_scores = sim_scores.sort_values(ascending=False)
    if top_n is not None:
        return sim_scores.head(top_n)
    return sim_scores


def _build_results(product_ids):
    results = []
    seen = set()
    for product in product_ids:
        if product in seen:
            continue
        seen.add(product)
        matching = df[df["product_id"] == product]
        if matching.empty:
            continue
        row = matching.iloc[0]
        results.append(
            {
                "product_id": product,
                "product_name": row["product_name"],
                "rating": row["rating"],
                "discounted_price": row["discounted_price"],
                "img_link": row["img_link"],
            }
        )
    return results


def get_hybrid_scores(
    user_id,
    product_id,
    top_n=200,
    exclude_items=None,
    user_history=None,
    cb_weight=0.70,
    cf_weight=0.30,
):
    user_id = str(user_id).strip()
    product_id = str(product_id).strip()
    rated_items = _coerce_history(user_id, user_history=user_history)

    cb_scores = get_cb_scores(product_id, top_n=top_n, exclude_items=exclude_items)
    cf_scores = get_cf_scores(
        user_id,
        top_k=max(top_n // 10, 20),
        exclude_items=exclude_items,
        user_history=rated_items,
    )
    profile_scores = _drop_excluded(_profile_content_scores(rated_items, top_n=top_n), exclude_items)

    candidates = cb_scores.index.union(cf_scores.index).union(profile_scores.index)
    if candidates.empty:
        return _get_popular_items(exclude_items).head(top_n)

    cb_norm = normalize_series(cb_scores.reindex(candidates).fillna(0))
    cf_norm = normalize_series(cf_scores.reindex(candidates).fillna(0))
    profile_norm = normalize_series(profile_scores.reindex(candidates).fillna(0))
    pop_norm = normalize_series(product_popularity.reindex(candidates).fillna(0))

    cf_nonzero = int((cf_norm > 0).sum())
    cf_confidence = min(1.0, cf_nonzero / max(top_n * 0.35, 1))
    activity_confidence = min(1.0, len(rated_items) / 4.0)
    confidence = 0.5 * cf_confidence + 0.5 * activity_confidence

    adaptive_cf = cf_weight * confidence
    profile_weight = cf_weight - adaptive_cf
    adaptive_cb = cb_weight
    pop_weight = 0.015 if confidence < 0.35 else 0.035

    total = adaptive_cb + adaptive_cf + profile_weight + pop_weight
    adaptive_cb /= total
    adaptive_cf /= total
    profile_weight /= total
    pop_weight /= total

    scores = (
        adaptive_cb * cb_norm
        + adaptive_cf * cf_norm
        + profile_weight * profile_norm
        + pop_weight * pop_norm
    )

    # Rank-fusion nudge preserves the 0.7/0.3 weighted score while helping sparse CF
    # candidates surface when score scales are close.
    if not cb_scores.empty:
        cb_rrf = 1.0 / (60 + cb_scores.rank(method="min", ascending=False))
        scores = scores.add(0.03 * cb_rrf.reindex(candidates).fillna(0), fill_value=0)
    if not cf_scores.empty and confidence > 0:
        cf_rrf = 1.0 / (60 + cf_scores.rank(method="min", ascending=False))
        scores = scores.add(0.05 * confidence * cf_rrf.reindex(candidates).fillna(0), fill_value=0)
    if not profile_scores.empty:
        profile_rrf = 1.0 / (60 + profile_scores.rank(method="min", ascending=False))
        scores = scores.add(0.04 * (1 - confidence) * profile_rrf.reindex(candidates).fillna(0), fill_value=0)

    scores = _drop_excluded(scores, exclude_items)
    return scores.sort_values(ascending=False).head(top_n)


def hybrid_recommend(user_id, product_id, top_n=5, exclude_items=None, user_history=None):
    user_id = str(user_id).strip()
    product_id = str(product_id).strip()
    rated_items = _coerce_history(user_id, user_history=user_history)

    scores = get_hybrid_scores(
        user_id,
        product_id,
        top_n=max(top_n * 40, 200),
        exclude_items=exclude_items,
        user_history=rated_items,
    )

    cb_scores = get_cb_scores(product_id, exclude_items=exclude_items)
    profile_scores = _drop_excluded(
        _profile_content_scores(rated_items, top_n=max(top_n * 40, 200)),
        exclude_items,
    )

    ranked_products = []
    for product in cb_scores.head(max(top_n - 1, 0)).index:
        if product not in ranked_products:
            ranked_products.append(product)

    for product in profile_scores.index:
        if len(ranked_products) >= top_n:
            break
        if product not in ranked_products:
            ranked_products.append(product)

    for product in scores.index:
        if len(ranked_products) >= top_n:
            break
        if product not in ranked_products:
            ranked_products.append(product)

    return _build_results(ranked_products[:top_n])


def hybrid_recommend_ranking_fusion(user_id, product_id, top_n=5):
    scores = get_hybrid_scores(user_id, product_id, top_n=max(top_n * 40, 200))
    return _build_results(scores.head(top_n).index.tolist())


def hybrid_recommend_contextual(user_id, product_id, top_n=5):
    scores = get_hybrid_scores(user_id, product_id, top_n=max(top_n * 40, 200))
    return _build_results(scores.head(top_n).index.tolist())
