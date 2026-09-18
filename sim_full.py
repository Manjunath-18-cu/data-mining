import pandas as pd, re, time
from sim import word_shingles, char_shingles, jaccard, normalize

notices = pd.read_parquet('notices.parquet').set_index('notice_id', drop=False)
pairs = pd.read_csv('labels.csv')

# precompute normalized shingle sets per notice referenced in labels
ids = pd.unique(pairs[['notice_id_a','notice_id_b']].values.ravel())
print("unique notices referenced in labelled_pairs:", len(ids))

body = {nid: notices.loc[nid].body for nid in ids}
norm = {nid: normalize(body[nid]) for nid in ids}
w5 = {nid: word_shingles(norm[nid], 5) for nid in ids}
c8 = {nid: char_shingles(norm[nid], 8) for nid in ids}

rows = []
for _, r in pairs.iterrows():
    a, b = r.notice_id_a, r.notice_id_b
    rows.append({
        'label': r.label,
        'word5_jaccard': jaccard(w5[a], w5[b]),
        'char8_jaccard': jaccard(c8[a], c8[b]),
    })
res = pd.DataFrame(rows)
res.to_csv('pair_scores.csv', index=False)

print("\n=== word 5-gram Jaccard by label ===")
print(res.groupby('label').word5_jaccard.describe())
print("\n=== char 8-gram Jaccard by label ===")
print(res.groupby('label').char8_jaccard.describe())

# separation quality: AUC-ish check via simple threshold sweep, and mean gap
import numpy as np
for col in ['word5_jaccard','char8_jaccard']:
    same_mean = res[res.label=='same'][col].mean()
    diff_mean = res[res.label=='different'][col].mean()
    # quick rank-based separation: what fraction of (same,diff) pairs are correctly ordered (same>diff)?
    s = res[res.label=='same'][col].values
    d = res[res.label=='different'][col].values
    import itertools
    # sample-based AUC estimate (full pairwise is 279*621=173k, fine)
    wins = (s[:,None] > d[None,:]).sum()
    ties = (s[:,None] == d[None,:]).sum()
    total = len(s)*len(d)
    auc = (wins + 0.5*ties) / total
    print(f"{col}: same_mean={same_mean:.4f} diff_mean={diff_mean:.4f} gap={same_mean-diff_mean:.4f} rank_AUC={auc:.4f}")
