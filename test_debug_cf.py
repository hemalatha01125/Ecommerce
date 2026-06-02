import pandas as pd
from app.recommender import get_cf_scores, get_cb_scores, user_item, user_means

user = list(user_item.index)[0]
print('Testing user:', user)
print('User means exists:', user_means.loc[user])
scores = get_cf_scores(user)
print('CF Scores length:', len(scores))
print('CF Scores sample:')
print(scores.head() if not scores.empty else 'EMPTY')
print('---')
print('Type:', type(scores))
