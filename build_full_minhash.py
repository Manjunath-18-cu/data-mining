import pickle, numpy as np, time, pandas as pd

with open('shingles.pkl','rb') as f:
    shingles = pickle.load(f)

NUM_PERM = 1068
MERSENNE = (1 << 61) - 1
rng = np.random.RandomState(42)
A = rng.randint(1, MERSENNE, size=NUM_PERM, dtype=np.int64)
B = rng.randint(0, MERSENNE, size=NUM_PERM, dtype=np.int64)

def hash_shingle(s):
    return int.from_bytes(hash(s).to_bytes(8,'little',signed=True),'little',signed=False) & ((1<<61)-1)

def minhash_sig(sset):
    if not sset: return np.full(NUM_PERM, MERSENNE, dtype=np.int64)
    h = np.array([hash_shingle(s) for s in sset], dtype=np.int64)
    perms = (np.outer(h, A) + B) % MERSENNE
    return perms.min(axis=0)

ids = list(shingles.keys())
t0=time.time()
sigs = np.zeros((len(ids), NUM_PERM), dtype=np.int64)
for i, nid in enumerate(ids):
    sigs[i] = minhash_sig(shingles[nid])
    if i % 3000 == 0: print(i, time.time()-t0)
t1=time.time()
print(f"built {len(ids)} signatures in {t1-t0:.1f}s  ({NUM_PERM} perms)")
np.save('sigs.npy', sigs)
with open('sig_ids.pkl','wb') as f:
    pickle.dump(ids, f)
