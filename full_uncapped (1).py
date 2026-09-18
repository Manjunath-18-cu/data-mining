import numpy as np, pickle, time, pandas as pd
from collections import defaultdict

sigs = np.load('sigs.npy')
with open('sig_ids.pkl','rb') as f: ids = pickle.load(f)
NUM_PERM, BANDS, ROWS = 1068, 267, 4

t0=time.time()
cand_pairs_uncapped = set()
for b in range(BANDS):
    cols = sigs[:, b*ROWS:(b+1)*ROWS]
    keys = [hash(tuple(row)) for row in cols]
    d = defaultdict(list)
    for i,k in enumerate(keys): d[k].append(i)
    for k, members in d.items():
        m = len(members)
        if m > 1:
            for i in range(m):
                for j in range(i+1, m):
                    a,b_ = members[i], members[j]
                    if a>b_: a,b_=b_,a
                    cand_pairs_uncapped.add((a,b_))
t1=time.time()
print(f"UNCAPPED candidate generation: {len(cand_pairs_uncapped)} distinct pairs in {t1-t0:.2f}s")
with open('cand_uncapped.pkl','wb') as f:
    pickle.dump(cand_pairs_uncapped, f)

# now estimate exact-scoring cost: time scoring a random sample of candidate pairs with exact jaccard, extrapolate
import random, sys
sys.path.insert(0,'.')
with open('shingles.pkl','rb') as f: shingles = pickle.load(f)
def jacc(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

sample = random.sample(list(cand_pairs_uncapped), 5000)
t2=time.time()
for i,j in sample:
    jacc(shingles[ids[i]], shingles[ids[j]])
t3=time.time()
per_pair = (t3-t2)/5000
print(f"exact-jaccard scoring: {per_pair*1e6:.1f} microseconds/pair (sampled)")
total_score_time_sec = per_pair * len(cand_pairs_uncapped)
print(f"extrapolated total exact-scoring time for {len(cand_pairs_uncapped)} candidates: {total_score_time_sec:.1f}s = {total_score_time_sec/60:.2f} min")
print(f"total nightly time (candidate-gen {t1-t0:.1f}s + scoring {total_score_time_sec:.1f}s) = {(t1-t0+total_score_time_sec)/60:.2f} min")
print(f"20-min budget: {'PASS' if (t1-t0+total_score_time_sec) <= 1200 else 'FAIL'}")
