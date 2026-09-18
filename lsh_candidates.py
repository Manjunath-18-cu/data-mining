import numpy as np, pickle, time, sqlite3, pandas as pd
from collections import defaultdict

sigs = np.load('sigs.npy')
with open('sig_ids.pkl','rb') as f: ids = pickle.load(f)
meta = pd.read_csv('notice_meta.csv').set_index('notice_id')

NUM_PERM = 1068
BANDS, ROWS = 267, 4
assert BANDS*ROWS == NUM_PERM

t0 = time.time()
# build band buckets
band_tables = []  # list of dict: bucket_key -> [row indices]
for b in range(BANDS):
    cols = sigs[:, b*ROWS:(b+1)*ROWS]
    # hash each row-window to a bucket key
    keys = [hash(tuple(row)) for row in cols]
    d = defaultdict(list)
    for i, k in enumerate(keys):
        d[k].append(i)
    band_tables.append(d)
t1 = time.time()
print(f"built {BANDS} band tables in {t1-t0:.2f}s")

# collect candidate pairs (dedup) + per-notice candidate counts + per-bucket sizes
t2 = time.time()
cand_pairs = set()
bucket_sizes = []
for d in band_tables:
    for k, members in d.items():
        m = len(members)
        if m > 1:
            bucket_sizes.append(m)
            if m <= 500:  # guard: skip pathological O(m^2) explosion for now, flag separately
                for i in range(m):
                    for j in range(i+1, m):
                        a, b_ = members[i], members[j]
                        if a > b_: a, b_ = b_, a
                        cand_pairs.add((a,b_))
t3 = time.time()
print(f"LSH candidate generation: {t3-t2:.2f}s, candidate pairs = {len(cand_pairs)}")
print(f"total possible pairs = {len(ids)*(len(ids)-1)//2}")
print(f"reduction factor = {len(ids)*(len(ids)-1)//2 / len(cand_pairs):.1f}x")

bucket_sizes = np.array(bucket_sizes)
print("\nbucket size distribution (buckets with >=2 members):")
print(pd.Series(bucket_sizes).describe())
print("bucket size 99th pct:", np.percentile(bucket_sizes,99), "max:", bucket_sizes.max())

with open('cand_pairs.pkl','wb') as f:
    pickle.dump({'cand_pairs':cand_pairs,'ids':ids,'bucket_sizes':bucket_sizes,
                 'band_build_time':t1-t0,'cand_gen_time':t3-t2}, f)
