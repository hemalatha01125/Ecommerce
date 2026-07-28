import hashlib

import pandas as pd

from .recommender import RAW_DATA_PATH, get_cb_scores, hybrid_recommend


PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "about_product",
    "img_link",
    "product_link",
]


def _load_products():
    raw = pd.read_csv(RAW_DATA_PATH)
    raw = raw.dropna(subset=["product_id", "product_name"])
    products = raw.loc[:, PRODUCT_COLUMNS].drop_duplicates(subset=["product_id"], keep="first")
    products["product_id"] = products["product_id"].astype(str).str.strip()
    products["rating"] = pd.to_numeric(products["rating"], errors="coerce")
    return products.reset_index(drop=True)


products_df = _load_products()


def _clean_scalar(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _stock_status(product_id):
    digest = hashlib.sha256(str(product_id).encode("utf-8")).hexdigest()
    return "Out of stock" if int(digest[:2], 16) % 11 == 0 else "In stock"


def _to_product_dict(row):
    rating = _clean_scalar(row.get("rating"))
    return {
        "product_id": _clean_scalar(row.get("product_id")),
        "title": _clean_scalar(row.get("product_name")),
        "description": _clean_scalar(row.get("about_product")),
        "category": _clean_scalar(row.get("category")),
        "price": _clean_scalar(row.get("discounted_price")),
        "actual_price": _clean_scalar(row.get("actual_price")),
        "discount_percentage": _clean_scalar(row.get("discount_percentage")),
        "rating": float(rating) if rating is not None else None,
        "rating_count": _clean_scalar(row.get("rating_count")),
        "image": _clean_scalar(row.get("img_link")),
        "product_link": _clean_scalar(row.get("product_link")),
        "stock_status": _stock_status(row.get("product_id")),
    }


def list_products(search=None, category=None, limit=24, offset=0):
    products = products_df.copy()

    if search:
        query = str(search).lower().strip()
        products = products[
            products["product_name"].astype(str).str.lower().str.contains(query, na=False)
            | products["about_product"].astype(str).str.lower().str.contains(query, na=False)
        ]

    if category:
        category_query = str(category).lower().strip()
        products = products[products["category"].astype(str).str.lower().str.contains(category_query, na=False)]

    total = len(products)
    products = products.iloc[offset : offset + limit]
    return {
        "items": [_to_product_dict(row) for _, row in products.iterrows()],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_product(product_id):
    matching = products_df[products_df["product_id"] == str(product_id).strip()]
    if matching.empty:
        return None
    return _to_product_dict(matching.iloc[0])


def get_categories(limit=50):
    categories = []
    seen = set()
    for value in products_df["category"].dropna().astype(str):
        first = value.split("|")[0].strip()
        if first and first not in seen:
            seen.add(first)
            categories.append(first)
        if len(categories) >= limit:
            break
    return categories


def get_similar_products(product_id, top_n=6):
    scores = get_cb_scores(product_id, top_n=top_n)
    items = []
    for pid in scores.index:
        product = get_product(pid)
        if product:
            items.append(product)
    return items


def get_personalized_recommendations(user_id, product_id, top_n=6):
    similar_ids = set(get_cb_scores(product_id, top_n=top_n).index)
    exclude_items = similar_ids | {str(product_id).strip()}
    rows = hybrid_recommend(user_id, product_id, top_n=top_n, exclude_items=exclude_items)
    items = []
    for row in rows:
        product = get_product(row["product_id"])
        if product:
            items.append(product)
    return items
