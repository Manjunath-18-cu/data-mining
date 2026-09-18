import pandas as pd, re

notices = pd.read_parquet('notices.parquet').set_index('notice_id', drop=False)

def word_shingles(text, k=5):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if len(toks) < k:
        return set(toks)
    return set(tuple(toks[i:i+k]) for i in range(len(toks)-k+1))

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def normalize(body):
    t = body
    # strip preamble boilerplate: keep from 'Name of work' onward if present
    m = re.search(r'Name of work', t)
    if m:
        t = t[m.start():]
    # strip disclaimer footer block
    t = re.split(r'-{20,}\s*\nDisclaimer:', t)[0]
    # canonicalize reference numbers (line 'Tender reference number: X.')
    t = re.sub(r'Tender reference number:\s*[^\n\.]+', 'Tender reference number: <REF>', t)
    # canonicalize money: Rs./INR amounts, bare digit amounts near currency-ish context, and known field lines
    money_fields = ['Estimated cost put to tender', 'Earnest money deposit', 'Cost of tender document',
                    'Minimum average annual turnover in the last three financial years',
                    'Experience of at least one similar completed work of value not less than']
    for f in money_fields:
        t = re.sub(re.escape(f) + r':\s*[^\n\.]+', f + ': <AMT>', t)
    # canonicalize dates: several raw formats
    date_pat = r'\b(\d{1,2}[-/\. ](?:\d{1,2}|[A-Za-z]{3,9})[-/\. ,]?\s*\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b'
    t = re.sub(date_pat, '<DATE>', t)
    return t

pairs = pd.read_csv('labels.csv')
same = pairs[pairs.label=='same'].iloc[0]
diff = pairs[pairs.label=='different'].iloc[0]

for tag, row in [('SAME (N010018,N010020)', same), ('DIFF (N007876,N008565)', diff)]:
    a = notices.loc[row.notice_id_a].body
    b = notices.loc[row.notice_id_b].body
    raw_j = jaccard(word_shingles(a), word_shingles(b))
    norm_a, norm_b = normalize(a), normalize(b)
    norm_j = jaccard(word_shingles(norm_a), word_shingles(norm_b))
    print(f"{tag}: raw_jaccard={raw_j:.4f}  normalized_jaccard={norm_j:.4f}  (norm_len_a={len(norm_a)}, norm_len_b={len(norm_b)})")

def char_shingles(text, k=8):
    t = re.sub(r'\s+', ' ', text.lower())
    if len(t) < k: return {t}
    return set(t[i:i+k] for i in range(len(t)-k+1))

print("\n--- char 8-gram shingles, same normalization pipeline ---")
for tag, row in [('SAME (N010018,N010020)', same), ('DIFF (N007876,N008565)', diff)]:
    a = notices.loc[row.notice_id_a].body
    b = notices.loc[row.notice_id_b].body
    raw_j = jaccard(char_shingles(a), char_shingles(b))
    norm_a, norm_b = normalize(a), normalize(b)
    norm_j = jaccard(char_shingles(norm_a), char_shingles(norm_b))
    print(f"{tag}: raw_char_jaccard={raw_j:.4f}  normalized_char_jaccard={norm_j:.4f}")
