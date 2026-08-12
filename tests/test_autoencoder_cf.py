import pandas as pd

from app.recommender import get_autoencoder_cf_scores


def test_autoencoder_cf_scores_are_generated():
    scores = get_autoencoder_cf_scores("user_0", top_k=5, exclude_items=set())
    assert isinstance(scores, pd.Series)
