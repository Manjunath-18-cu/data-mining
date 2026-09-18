import numpy as np, pickle, time, pandas as pd
from collections import defaultdict

sigs = np.load('sigs.npy')
with open('sig_ids.pkl','rb') as f: ids = pickle.load(f)
notices = pd.read_parquet('notices.parquet').set_index('notice_id')
NUM_PERM, BANDS, ROWS = 1068, 267, 4

# secondary blocking key: estimated_value rounded to nearest 100,000
sec_key = (notices.loc[ids].estimated_value.values // 100000).astype(np.int64)

t0=time.time()
cand_pairs_mit = set()
for b in range(BANDS):
    cols = sigs[:, b*ROWS:(b+1)*ROWS]
    keys=[hash(tuple(row)) for row in cols]
    d=defaultdict(list)
    for i,k in enumerate(keys): d[k].append(i)
    for k, members in d.items():
        if len(members) <=1: continue
        # sub-partition by secondary key
        sub = defaultdict(list)
        for i in members:
            sub[sec_key[i]].append(i)
        for skey, smembers in sub.items():
            m = len(smembers)
            if m>1:
                for x in range(m):
                    for y in range(x+1,m):
                        a,b_ = smembers[x], smembers[y]
                        if a>b_: a,b_=b_,a
                        cand_pairs_mit.add((a,b_))
t1=time.time()
gen_time = t1-t0
print(f"MITIGATED candidate generation: {len(cand_pairs_mit)} distinct pairs in {gen_time:.2f}s")

with open('shingles.pkl','rb') as f: shingles = pickle.load(f)
def jacc(a,b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)
import random
sample = random.sample(list(cand_pairs_mit), min(5000,len(cand_pairs_mit)))
t2=time.time()
for i,j in sample: jacc(shingles[ids[i]], shingles[ids[j]])
t3=time.time()
per_pair=(t3-t2)/len(sample)
total_score_time = per_pair*len(cand_pairs_mit)
total_time = gen_time+total_score_time
print(f"exact-scoring extrapolated: {total_score_time:.1f}s, total nightly time = {total_time:.1f}s = {total_time/60:.2f} min")
print(f"20-min budget: {'PASS' if total_time<=1200 else 'FAIL'}")

with open('cand_mitigated.pkl','wb') as f:
    pickle.dump({'cand_pairs':cand_pairs_mit,'gen_time':gen_time,'total_time':total_time}, f)

# ---- recall check on labelled_pairs.csv: are same-pairs still generated as candidates? ----
labels = pd.read_csv('labels.csv')
id_to_idx = {nid:i for i,nid in enumerate(ids)}
missed=0; checked=0
for _,r in labels[labels.label=='same'].iterrows():
    if r.notice_id_a not in id_to_idx or r.notice_id_b not in id_to_idx: continue
    checked+=1
    a,b_ = id_to_idx[r.notice_id_a], id_to_idx[r.notice_id_b]
    if a>b_: a,b_=b_,a
    if (a,b_) not in cand_pairs_mit:
        missed+=1
print(f"\nrecall check (mitigated): {checked-missed}/{checked} labeled SAME pairs still recovered as candidates ({100*(checked-missed)/checked:.2f}%)")

# same check on the UNCAPPED (pre-mitigation) set for comparison
with open('cand_uncapped.pkl','rb') as f: cand_uncapped = pickle.load(f)
missed_u=0
for _,r in labels[labels.label=='same'].iterrows():
    if r.notice_id_a not in id_to_idx or r.notice_id_b not in id_to_idx: continue
    a,b_ = id_to_idx[r.notice_id_a], id_to_idx[r.notice_id_b]
    if a>b_: a,b_=b_,a
    if (a,b_) not in cand_uncapped:
        missed_u+=1
print(f"recall check (pre-mitigation/uncapped): {checked-missed_u}/{checked} recovered ({100*(checked-missed_u)/checked:.2f}%)")
