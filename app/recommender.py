import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


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


df = None
interaction_df = None
product_id_to_idx = {}
tfidf = None
tfidf_matrix = None
content_sim = None

AUTOENCODER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "autoencoder_cf_model.pkl")
AUTOENCODER_METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "autoencoder_cf_metadata.pkl")
BEHAVIOR_CF_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "behavior_cf_model.pkl")
BEHAVIOR_CF_METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "behavior_cf_metadata.pkl")
MIN_BEHAVIOR_EVENTS_FOR_RETRAIN = 12

EVENT_SCORE_WEIGHTS = {
    "view": 2.0,
    "like": 3.0,
    "wishlist": 3.5,
    "rating": 4.0,
    "cart": 4.2,
    "purchase": 5.0,
}


# =========================
# AUTOENCODER COLLABORATIVE FILTERING
# =========================

def ensure_data_loaded():
    """Load catalog and interaction data lazily so the module imports quickly."""
    global df, interaction_df, product_id_to_idx
    if df is None or interaction_df is None:
        df, interaction_df = _load_catalog_and_interactions()
        product_id_to_idx.clear()
        product_id_to_idx.update(pd.Series(df.index, index=df["product_id"]).to_dict())
    return df, interaction_df


def behavior_user_id(user_id):
    return f"auth:{str(user_id).strip()}"


def _interaction_with_timestamps(interactions):
    dated = interactions.copy()
    dated["event_type"] = "catalog_rating"
    dated["source"] = "catalog"
    dated["created_at"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(dated)), unit="s")
    return dated[["user_id", "product_id", "rating", "event_type", "source", "created_at"]]


def _score_behavior_event(event_type, score):
    event_type = str(event_type).strip().lower()
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = EVENT_SCORE_WEIGHTS.get(event_type, 1.0)

    if event_type == "rating":
        return float(np.clip(score, 1.0, 5.0))
    return float(np.clip(1.0 + 0.85 * np.log1p(score), 1.0, 5.0))


def load_behavior_interactions():
    """Read authenticated behavior events as separate user-item interactions."""
    try:
        from flask import has_app_context
        if not has_app_context():
            return pd.DataFrame(columns=["user_id", "product_id", "rating", "event_type", "source", "created_at"])
        from .models import UserBehaviorEvent

        rows = UserBehaviorEvent.query.order_by(UserBehaviorEvent.created_at.asc()).all()
    except Exception:
        return pd.DataFrame(columns=["user_id", "product_id", "rating", "event_type", "source", "created_at"])

    records = []
    for row in rows:
        product_id = str(row.product_id).strip()
        if not product_id:
            continue
        records.append(
            {
                "user_id": behavior_user_id(row.user_id),
                "product_id": product_id,
                "rating": _score_behavior_event(row.event_type, row.score),
                "event_type": str(row.event_type).strip().lower(),
                "source": "behavior",
                "created_at": pd.to_datetime(row.created_at),
            }
        )

    return pd.DataFrame(records, columns=["user_id", "product_id", "rating", "event_type", "source", "created_at"])


def get_interaction_data(include_behavior=False, behavior_df=None):
    """Return catalog ratings alone or catalog ratings plus authenticated behavior."""
    _, base_interactions = ensure_data_loaded()
    base = _interaction_with_timestamps(base_interactions)
    if not include_behavior:
        return base

    behavior = load_behavior_interactions() if behavior_df is None else behavior_df.copy()
    if behavior.empty:
        return base

    behavior["user_id"] = behavior["user_id"].astype(str).str.strip()
    behavior["product_id"] = behavior["product_id"].astype(str).str.strip()
    behavior["rating"] = pd.to_numeric(behavior["rating"], errors="coerce")
    behavior["created_at"] = pd.to_datetime(behavior["created_at"], errors="coerce")
    behavior = behavior.dropna(subset=["user_id", "product_id", "rating", "created_at"])
    behavior = behavior[behavior["product_id"].isin(df["product_id"])]
    return pd.concat([base, behavior[base.columns]], ignore_index=True)


def ensure_content_features():
    """Build content-based features lazily when content-based recommendations are requested."""
    global tfidf, tfidf_matrix, content_sim
    if tfidf is not None and content_sim is not None:
        return tfidf, tfidf_matrix, content_sim

    catalog, _ = ensure_data_loaded()
    catalog = catalog.copy()
    catalog["brand"] = catalog["product_name"].astype(str).str.extract(r"^\s*([A-Za-z0-9]+)", expand=False).fillna("")

    for col in ["product_name", "category", "about_product", "brand"]:
        catalog[f"clean_{col}"] = catalog[col].astype(str).apply(clean_text)

    catalog["content"] = (
        (catalog["clean_product_name"] + " ") * 3
        + (catalog["clean_category"] + " ") * 3
        + (catalog["clean_brand"] + " ") * 2
        + catalog["clean_about_product"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = tfidf.fit_transform(catalog["content"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, content_sim


def _build_user_item_matrix(interactions, user_col="user_id", item_col="product_id", rating_col="rating", max_users=1500, max_items=2000):
    """Build a dense user-item matrix for autoencoder training from a compact active subset."""
    if interactions.empty:
        return pd.DataFrame()
    matrix = interactions.pivot_table(index=user_col, columns=item_col, values=rating_col, aggfunc="mean")
    matrix = matrix.fillna(0.0)

    if max_users is not None and len(matrix.index) > max_users:
        user_counts = interactions.groupby(user_col).size().sort_values(ascending=False)
        active_users = user_counts.head(max_users).index.tolist()
        matrix = matrix.loc[matrix.index.isin(active_users)]

    if max_items is not None and len(matrix.columns) > max_items:
        item_counts = interactions.groupby(item_col).size().sort_values(ascending=False)
        active_items = item_counts.head(max_items).index.tolist()
        matrix = matrix.loc[:, matrix.columns.isin(active_items)]

    return matrix


def _prepare_autoencoder_training_data(matrix):
    """Scale and return training arrays for the autoencoder."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaled = scaler.fit_transform(matrix.values)
    return scaled, scaler


class AutoencoderCF:
    """A lightweight NumPy autoencoder for latent collaborative filtering."""

    def __init__(self, input_dim, latent_dim=16, dropout=0.2, learning_rate=1e-3, random_state=42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.scaler = None
        self.user_ids = None
        self.item_ids = None

    @staticmethod
    def _sigmoid(values):
        return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))

    def _init_weights(self, input_dim, latent_dim):
        rng = np.random.default_rng(self.random_state)
        scale = 1.0 / np.sqrt(max(input_dim, 1))
        encoder_w = rng.normal(0, scale, size=(input_dim, latent_dim))
        encoder_b = np.zeros(latent_dim)
        decoder_w = rng.normal(0, scale, size=(latent_dim, input_dim))
        decoder_b = np.zeros(input_dim)
        return encoder_w, encoder_b, decoder_w, decoder_b

    def fit(self, matrix, epochs=4, batch_size=128, validation_split=0.1):
        self.user_ids = matrix.index.tolist()
        self.item_ids = matrix.columns.tolist()
        scaled, scaler = _prepare_autoencoder_training_data(matrix)
        self.scaler = scaler
        encoder_w, encoder_b, decoder_w, decoder_b = self._init_weights(scaled.shape[1], self.latent_dim)
        self.weights = {
            "encoder_w": encoder_w,
            "encoder_b": encoder_b,
            "decoder_w": decoder_w,
            "decoder_b": decoder_b,
        }

        n_samples = scaled.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            rng = np.random.default_rng(self.random_state + epoch)
            rng.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch = scaled[batch_idx]
                hidden_pre = batch @ encoder_w + encoder_b
                hidden = np.maximum(hidden_pre, 0)
                output = self._sigmoid(hidden @ decoder_w + decoder_b)
                error = output - batch
                output_grad = error * output * (1.0 - output)
                hidden_grad = output_grad @ decoder_w.T * (hidden_pre > 0)
                decoder_w_grad = hidden.T @ output_grad / max(len(batch_idx), 1)
                decoder_b_grad = output_grad.mean(axis=0)
                encoder_w_grad = batch.T @ hidden_grad / max(len(batch_idx), 1)
                encoder_b_grad = hidden_grad.mean(axis=0)
                decoder_w -= self.learning_rate * decoder_w_grad
                decoder_b -= self.learning_rate * decoder_b_grad
                encoder_w -= self.learning_rate * encoder_w_grad
                encoder_b -= self.learning_rate * encoder_b_grad

        return self

    def predict_scores(self, matrix):
        if self.weights is None or self.scaler is None:
            raise ValueError("Model is not trained")
        scaled = self.scaler.transform(matrix.values)
        encoder_w = self.weights["encoder_w"]
        encoder_b = self.weights["encoder_b"]
        decoder_w = self.weights["decoder_w"]
        decoder_b = self.weights["decoder_b"]
        hidden = np.maximum(scaled @ encoder_w + encoder_b, 0)
        preds = self._sigmoid(hidden @ decoder_w + decoder_b)
        return self.scaler.inverse_transform(preds)

    def save(self, path, metadata_path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        metadata = {
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
        }
        with open(metadata_path, "wb") as handle:
            pickle.dump(metadata, handle)

    @classmethod
    def load(cls, path, metadata_path):
        try:
            with open(path, "rb") as handle:
                model = pickle.load(handle)
            with open(metadata_path, "rb") as handle:
                metadata = pickle.load(handle)
        except Exception:
            return None

        if not isinstance(model, cls):
            return None

        if isinstance(metadata, dict):
            model.user_ids = metadata.get("user_ids")
            model.item_ids = metadata.get("item_ids")
        else:
            model.user_ids = getattr(metadata, "user_ids", None)
            model.item_ids = getattr(metadata, "item_ids", None)
        return model


autoencoder_cf_model = None
autoencoder_cf_metadata = None


def _train_autoencoder_cf_if_needed():
    global autoencoder_cf_model, autoencoder_cf_metadata
    if autoencoder_cf_model is not None:
        return autoencoder_cf_model

    if os.path.exists(AUTOENCODER_MODEL_PATH) and os.path.exists(AUTOENCODER_METADATA_PATH):
        loaded_model = AutoencoderCF.load(AUTOENCODER_MODEL_PATH, AUTOENCODER_METADATA_PATH)
        if loaded_model is not None:
            autoencoder_cf_model = loaded_model
            autoencoder_cf_metadata = {
                "user_ids": autoencoder_cf_model.user_ids,
                "item_ids": autoencoder_cf_model.item_ids,
            }
            return autoencoder_cf_model

    _, interactions = ensure_data_loaded()
    matrix = _build_user_item_matrix(interactions[["user_id", "product_id", "rating"]], max_users=1000, max_items=1200)
    if matrix.empty:
        return None
    model = AutoencoderCF(input_dim=matrix.shape[1], latent_dim=16)
    model.fit(matrix, epochs=3, batch_size=128)
    model.save(AUTOENCODER_MODEL_PATH, AUTOENCODER_METADATA_PATH)
    autoencoder_cf_model = model
    autoencoder_cf_metadata = {
        "user_ids": model.user_ids,
        "item_ids": model.item_ids,
    }
    return model


def get_autoencoder_cf_scores(user_id, top_k=15, exclude_items=None, user_history=None, include_fallback=True):
    """Generate collaborative scores from the trained autoencoder model."""
    user_id = str(user_id).strip()
    model = _train_autoencoder_cf_if_needed()
    if model is None:
        return _get_popular_items(exclude_items) if include_fallback else pd.Series(dtype=float)

    _, interactions = ensure_data_loaded()
    matrix = _build_user_item_matrix(interactions[["user_id", "product_id", "rating"]], max_users=1000, max_items=1200)
    if user_id not in matrix.index:
        return _get_popular_items(exclude_items) if include_fallback else pd.Series(dtype=float)

    user_row = matrix.loc[[user_id]].copy()
    if user_row.empty:
        return _get_popular_items(exclude_items) if include_fallback else pd.Series(dtype=float)

    pred_matrix = pd.DataFrame(model.predict_scores(matrix), index=matrix.index, columns=matrix.columns)
    user_scores = pred_matrix.loc[user_id]
    user_scores = user_scores.astype(float)
    if user_history is not None:
        history_index = list(pd.Series(user_history).index)
        user_scores = user_scores.drop(labels=history_index, errors="ignore")
    user_scores = _drop_excluded(user_scores, exclude_items)
    scores = user_scores[user_scores > 0].sort_values(ascending=False)
    if include_fallback and len(scores) < top_k:
        popularity = normalize_series(_get_popular_items(exclude_items))
        scores = scores.combine_first(popularity)
    return scores.head(top_k).sort_values(ascending=False)


# =========================
# CONTENT-BASED FILTERING
# =========================

# =========================
# COLLABORATIVE FILTERING
# =========================

_, interactions = ensure_data_loaded()
user_item = interactions.pivot_table(
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

product_stats = interactions.groupby("product_id")["rating"].agg(["mean", "count"])
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
    ensure_data_loaded()
    ensure_content_features()
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


def get_legacy_cf_scores(
    user_id,
    top_k=15,
    min_similarity=0.0,
    exclude_items=None,
    user_history=None,
    include_fallback=True,
):
    """Legacy item-user KNN collaborative filtering used as a baseline for comparison."""
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


def get_cf_scores(
    user_id,
    top_k=15,
    min_similarity=0.0,
    exclude_items=None,
    user_history=None,
    include_fallback=True,
):
    """Collaborative scores using the autoencoder-based CF model with legacy KNN fallback."""
    user_id = str(user_id).strip()
    rated_items = _coerce_history(user_id, user_history=user_history)

    if rated_items.empty:
        return _get_popular_items(exclude_items) if include_fallback else pd.Series(dtype=float)

    if exclude_items is None:
        exclude_items = set(rated_items.index)
    else:
        exclude_items = set(exclude_items)

    autoencoder_scores = get_autoencoder_cf_scores(
        user_id,
        top_k=top_k,
        exclude_items=exclude_items,
        user_history=rated_items,
        include_fallback=False,
    )
    if not autoencoder_scores.empty:
        scores = autoencoder_scores
    else:
        scores = get_legacy_cf_scores(
            user_id,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_items=exclude_items,
            user_history=rated_items,
            include_fallback=include_fallback,
        )

    scores = _drop_excluded(scores, exclude_items)
    scores = scores[scores > 0].sort_values(ascending=False)

    if include_fallback and len(scores) < top_k:
        popularity = normalize_series(_get_popular_items(exclude_items))
        scores = scores.combine_first(popularity)

    return scores.sort_values(ascending=False)


def _dynamic_item_knn_scores(rated_items, matrix, k_neighbors=80, min_similarity=0.0):
    if rated_items.empty or matrix.empty:
        return pd.Series(dtype=float)

    item_means_local = matrix.replace(0, np.nan).mean(axis=0)
    item_user_centered_local = matrix.replace(0, np.nan).sub(item_means_local, axis=1).fillna(0).T
    if item_user_centered_local.empty:
        return pd.Series(dtype=float)

    sim = cosine_similarity(item_user_centered_local)
    sim_df = pd.DataFrame(sim, index=item_user_centered_local.index, columns=item_user_centered_local.index)

    scores = pd.Series(dtype=float)
    total_weight = pd.Series(dtype=float)
    for item_id, rating in rated_items.items():
        if item_id not in sim_df.index:
            continue
        sims = sim_df.loc[item_id].drop(labels=[item_id], errors="ignore")
        sims = sims[sims > min_similarity].nlargest(k_neighbors)
        if sims.empty:
            continue
        rating_weight = max(float(rating) - 3.0, 0.25)
        scores = scores.add(sims * rating_weight, fill_value=0)
        total_weight = total_weight.add(sims.abs(), fill_value=0)

    if scores.empty:
        return scores
    return (scores / total_weight.replace(0, np.nan)).dropna()


def _dynamic_user_knn_scores(user_id, rated_items, matrix, k_neighbors=60, min_similarity=0.0):
    if user_id not in matrix.index or matrix.empty:
        return pd.Series(dtype=float)

    user_means_local = matrix.replace(0, np.nan).mean(axis=1).fillna(0)
    centered = matrix.sub(user_means_local, axis=0).fillna(0)
    sim = cosine_similarity(centered)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    sims = sim_df.loc[user_id].drop(labels=[user_id], errors="ignore")
    sims = sims[sims > min_similarity].nlargest(k_neighbors)
    if sims.empty:
        return pd.Series(dtype=float)

    neighbor_ratings = matrix.loc[sims.index].sub(user_means_local.loc[sims.index], axis=0).fillna(0)
    weighted = neighbor_ratings.mul(sims, axis=0).sum(axis=0)
    weights = neighbor_ratings.ne(0).mul(sims.abs(), axis=0).sum(axis=0).replace(0, np.nan)
    scores = (weighted / weights).dropna() + rated_items.mean()
    return scores[scores > 0]


def get_behavior_cf_scores(
    user_id,
    top_k=15,
    min_similarity=0.0,
    exclude_items=None,
    user_history=None,
    behavior_df=None,
    include_fallback=True,
):
    """Collaborative filtering over catalog ratings plus real authenticated behavior."""
    interactions = get_interaction_data(include_behavior=True, behavior_df=behavior_df)
    matrix = _build_user_item_matrix(interactions[["user_id", "product_id", "rating"]], max_users=1800, max_items=2200)
    rated_items = _coerce_history(user_id, user_history=user_history)

    if rated_items.empty and user_id in matrix.index:
        rated_items = matrix.loc[user_id]
        rated_items = rated_items[rated_items > 0]

    if rated_items.empty:
        return _get_popular_items(exclude_items).head(top_k) if include_fallback else pd.Series(dtype=float)

    if exclude_items is None:
        exclude_items = set(rated_items.index)
    else:
        exclude_items = set(exclude_items)

    auto_scores = pd.Series(dtype=float)
    model = AutoencoderCF.load(BEHAVIOR_CF_MODEL_PATH, BEHAVIOR_CF_METADATA_PATH)
    if model is not None and user_id in matrix.index and model.item_ids:
        try:
            aligned_matrix = matrix.reindex(columns=model.item_ids, fill_value=0.0)
            pred_matrix = pd.DataFrame(model.predict_scores(aligned_matrix), index=aligned_matrix.index, columns=aligned_matrix.columns)
            auto_scores = pred_matrix.loc[user_id].astype(float)
        except Exception:
            auto_scores = pd.Series(dtype=float)

    item_scores = _dynamic_item_knn_scores(rated_items, matrix, k_neighbors=max(top_k * 8, 80), min_similarity=min_similarity)
    user_scores = _dynamic_user_knn_scores(user_id, rated_items, matrix, k_neighbors=max(top_k * 5, 50), min_similarity=min_similarity)
    profile_scores = _profile_content_scores(rated_items, top_n=max(top_k * 20, 200))

    candidates = item_scores.index.union(user_scores.index).union(profile_scores.index).union(auto_scores.index)
    if candidates.empty:
        scores = pd.Series(dtype=float)
    else:
        auto_norm = normalize_series(auto_scores.reindex(candidates).fillna(0))
        item_norm = normalize_series(item_scores.reindex(candidates).fillna(0))
        user_norm = normalize_series(user_scores.reindex(candidates).fillna(0))
        profile_norm = normalize_series(profile_scores.reindex(candidates).fillna(0))
        behavior_confidence = min(1.0, len(rated_items) / 6.0)
        scores = (
            (0.30 + 0.10 * behavior_confidence) * item_norm
            + (0.20 + 0.10 * behavior_confidence) * user_norm
            + 0.20 * auto_norm
            + (0.30 - 0.20 * behavior_confidence) * profile_norm
        )

    scores = _drop_excluded(scores, exclude_items)
    scores = scores[scores > 0].sort_values(ascending=False)
    if include_fallback and len(scores) < top_k:
        popularity = normalize_series(_get_popular_items(exclude_items))
        scores = scores.combine_first(popularity)
    return scores.sort_values(ascending=False).head(top_k)


def maybe_retrain_behavior_cf(behavior_df=None, force=False):
    """Refresh the behavior-aware CF artifact after enough new behavior exists."""
    behavior = load_behavior_interactions() if behavior_df is None else behavior_df.copy()
    trained_events = 0
    if os.path.exists(BEHAVIOR_CF_METADATA_PATH):
        try:
            with open(BEHAVIOR_CF_METADATA_PATH, "rb") as handle:
                metadata = pickle.load(handle)
            if isinstance(metadata, dict):
                trained_events = int(metadata.get("trained_events", 0))
        except Exception:
            trained_events = 0
    if not force:
        if len(behavior) < MIN_BEHAVIOR_EVENTS_FOR_RETRAIN:
            return False
        if len(behavior) - trained_events < MIN_BEHAVIOR_EVENTS_FOR_RETRAIN:
            return False

    interactions = get_interaction_data(include_behavior=True, behavior_df=behavior)
    matrix = _build_user_item_matrix(interactions[["user_id", "product_id", "rating"]], max_users=1800, max_items=2200)
    if matrix.empty:
        return False

    model = AutoencoderCF(input_dim=matrix.shape[1], latent_dim=20)
    model.fit(matrix, epochs=3, batch_size=128)
    model.save(BEHAVIOR_CF_MODEL_PATH, BEHAVIOR_CF_METADATA_PATH)
    try:
        with open(BEHAVIOR_CF_METADATA_PATH, "rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["trained_events"] = len(behavior)
        with open(BEHAVIOR_CF_METADATA_PATH, "wb") as handle:
            pickle.dump(metadata, handle)
    except Exception:
        pass
    return True


# =========================
# CONTENT-BASED FILTERING
# =========================

def get_cb_scores(product_id, top_n=None, exclude_items=None):
    ensure_data_loaded()
    ensure_content_features()
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
    ensure_data_loaded()
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


def get_behavior_hybrid_scores(
    user_id,
    product_id,
    top_n=200,
    exclude_items=None,
    user_history=None,
    behavior_df=None,
):
    user_id = str(user_id).strip()
    product_id = str(product_id).strip()
    rated_items = _coerce_history(user_id, user_history=user_history)

    cb_scores = get_cb_scores(product_id, top_n=top_n, exclude_items=exclude_items)
    behavior_cf_scores = get_behavior_cf_scores(
        user_id,
        top_k=max(top_n // 5, 40),
        exclude_items=exclude_items,
        user_history=rated_items,
        behavior_df=behavior_df,
        include_fallback=False,
    )
    profile_scores = _drop_excluded(_profile_content_scores(rated_items, top_n=top_n), exclude_items)

    candidates = cb_scores.index.union(behavior_cf_scores.index).union(profile_scores.index)
    if candidates.empty:
        return _get_popular_items(exclude_items).head(top_n)

    cb_norm = normalize_series(cb_scores.reindex(candidates).fillna(0))
    cf_norm = normalize_series(behavior_cf_scores.reindex(candidates).fillna(0))
    profile_norm = normalize_series(profile_scores.reindex(candidates).fillna(0))
    pop_norm = normalize_series(product_popularity.reindex(candidates).fillna(0))

    activity_confidence = min(1.0, len(rated_items) / 5.0)
    cf_confidence = min(1.0, int((cf_norm > 0).sum()) / max(top_n * 0.25, 1))
    confidence = 0.65 * activity_confidence + 0.35 * cf_confidence

    cb_weight = 0.45 - 0.10 * confidence
    cf_weight = 0.20 + 0.25 * confidence
    profile_weight = 0.30 + 0.05 * activity_confidence
    pop_weight = 0.05
    total = cb_weight + cf_weight + profile_weight + pop_weight

    scores = (
        (cb_weight / total) * cb_norm
        + (cf_weight / total) * cf_norm
        + (profile_weight / total) * profile_norm
        + (pop_weight / total) * pop_norm
    )

    if not behavior_cf_scores.empty:
        cf_rrf = 1.0 / (50 + behavior_cf_scores.rank(method="min", ascending=False))
        scores = scores.add(0.08 * confidence * cf_rrf.reindex(candidates).fillna(0), fill_value=0)
    if not profile_scores.empty:
        profile_rrf = 1.0 / (55 + profile_scores.rank(method="min", ascending=False))
        scores = scores.add(0.05 * activity_confidence * profile_rrf.reindex(candidates).fillna(0), fill_value=0)

    return _drop_excluded(scores, exclude_items).sort_values(ascending=False).head(top_n)


def behavior_hybrid_recommend(user_id, product_id, top_n=5, exclude_items=None, user_history=None, behavior_df=None):
    rated_items = _coerce_history(user_id, user_history=user_history)
    scores = get_behavior_hybrid_scores(
        user_id,
        product_id,
        top_n=max(top_n * 40, 200),
        exclude_items=exclude_items,
        user_history=rated_items,
        behavior_df=behavior_df,
    )

    ranked_products = []
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
