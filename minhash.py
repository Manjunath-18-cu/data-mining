import pandas as pd, numpy as np, re, time
from sim import word_shingles, normalize, jaccard

notices = pd.read_parquet('notices.parquet').set_index('notice_id', drop=False)
pairs = pd.read_csv('labels.csv')
ids = pd.unique(pairs[['notice_id_a','notice_id_b']].values.ravel())

norm = {nid: normalize(notices.loc[nid].body) for nid in ids}
shingles = {nid: word_shingles(norm[nid], 5) for nid in ids}

NUM_PERM = 1068
MERSENNE = (1 << 61) - 1
rng = np.random.RandomState(42)
A = rng.randint(1, MERSENNE, size=NUM_PERM, dtype=np.int64)
B = rng.randint(0, MERSENNE, size=NUM_PERM, dtype=np.int64)

def hash_shingle(s):
    return int.from_bytes(hash(s).to_bytes(8, 'little', signed=True), 'little', signed=False) & ((1<<61)-1)

def minhash_sig(shingle_set):
    if not shingle_set:
        return np.full(NUM_PERM, MERSENNE, dtype=np.int64)
    h = np.array([hash_shingle(s) for s in shingle_set], dtype=np.int64)
    # (A*h + B) mod MERSENNE, vectorized: shape (num_shingles, num_perm)
    perms = (np.outer(h, A) + B) % MERSENNE
    return perms.min(axis=0)

t0 = time.time()
sigs = {nid: minhash_sig(shingles[nid]) for nid in ids}
t1 = time.time()
print(f"built {len(sigs)} signatures (num_perm={NUM_PERM}) in {t1-t0:.2f}s")

rows = []
for _, r in pairs.iterrows():
    a, b = r.notice_id_a, r.notice_id_b
    est = float(np.mean(sigs[a] == sigs[b]))
    exact = jaccard(shingles[a], shingles[b])
    rows.append({'label': r.label, 'exact': exact, 'estimated': est, 'abs_error': abs(exact-est)})
res = pd.DataFrame(rows)
res.to_csv('minhash_error.csv', index=False)
print(res.abs_error.describe())
print("MAE:", res.abs_error.mean())
print("median AE:", res.abs_error.median())
print("95th pct AE:", res.abs_error.quantile(0.95))
print("max AE:", res.abs_error.max())
print("\nworst 5 cases:")
print(res.sort_values('abs_error', ascending=False).head(5))
