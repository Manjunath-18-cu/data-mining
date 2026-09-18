import numpy as np, pickle, time, pandas as pd
from collections import defaultdict

sigs = np.load('sigs.npy')
with open('sig_ids.pkl','rb') as f: ids = pickle.load(f)
meta = pd.read_csv('notice_meta.csv').set_index('notice_id')

NUM_PERM, BANDS, ROWS = 1068, 267, 4

# find the largest bucket overall (unmitigated view) and inspect it
worst = None
for b in range(BANDS):
    cols = sigs[:, b*ROWS:(b+1)*ROWS]
    keys = [hash(tuple(row)) for row in cols]
    d = defaultdict(list)
    for i,k in enumerate(keys): d[k].append(i)
    for k, members in d.items():
        if worst is None or len(members) > len(worst[1]):
            worst = (b, members)

b, members = worst
print(f"largest bucket: band={b}, size={len(members)}")
notice_ids_in_bucket = [ids[i] for i in members]
portals = meta.loc[notice_ids_in_bucket].portal_id.value_counts()
print("portal composition of largest bucket:")
print(portals.head(15))
print("nodal portals P001-P006 share of this bucket:",
      meta.loc[notice_ids_in_bucket].portal_id.isin(['P001','P002','P003','P004','P005','P006']).mean())

# also: unmitigated TRUE total pairwise cost if we didn't cap bucket size
total_uncapped_pairs = 0
bucket_size_all = []
for b2 in range(BANDS):
    cols = sigs[:, b2*ROWS:(b2+1)*ROWS]
    keys = [hash(tuple(row)) for row in cols]
    d = defaultdict(list)
    for i,k in enumerate(keys): d[k].append(i)
    for k, mem in d.items():
        m = len(mem)
        if m > 1:
            bucket_size_all.append(m)
            total_uncapped_pairs += m*(m-1)//2
print("\nTRUE uncapped total (band,bucket) pair-work sum (with duplication across bands):", total_uncapped_pairs)
bucket_size_all = np.array(bucket_size_all)
top1pct_n = max(1,int(len(bucket_size_all)*0.01))
sorted_desc = np.sort(bucket_size_all)[::-1]
work_per_bucket = sorted_desc*(sorted_desc-1)//2
top1pct_work = work_per_bucket[:top1pct_n].sum()
print(f"buckets total={len(bucket_size_all)}, top 1% ({top1pct_n} buckets) responsible for {top1pct_work} of {total_uncapped_pairs} pair-work units = {100*top1pct_work/total_uncapped_pairs:.1f}%")
