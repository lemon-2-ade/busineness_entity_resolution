"""Independent sanity checks of the two submission files (mirrors the rules of
utils/validate_submission.py; run that official validator as well).
python scripts/check_outputs.py --out output --test-dir data/test"""
import argparse, sys

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='output'); ap.add_argument('--test-dir', default='data/test')
a = ap.parse_args()


def ids(path):
    with open(path) as f:
        next(f)
        return [line.split('\t', 1)[0] for line in f]


s1 = ids(f'{a.test_dir}/test_source1.tsv')
s23 = set(ids(f'{a.test_dir}/test_source2.tsv')) | set(ids(f'{a.test_dir}/test_source3.tsv'))
errors = []


def read(path, col):
    rows = {}
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        if head != ['source1_entity_id', col]:
            errors.append(f'{path}: bad header {head}')
        for line in f:
            k, v = line.rstrip('\n').split('\t')
            if k in rows:
                errors.append(f'{path}: duplicate row {k}')
            lst = v.split(',') if v else []
            if len(set(lst)) != len(lst):
                errors.append(f'{path}: duplicate ids in {k}')
            rows[k] = lst
    return rows


m = read(f'{a.out}/matching_results.tsv', 'matched_entity_ids')
c = read(f'{a.out}/candidate_pairs.tsv', 'candidate_entity_ids')
for name, d in (('matching', m), ('candidate', c)):
    if set(d) != set(s1) or len(d) != len(s1):
        errors.append(f'{name}: S1 rows mismatch ({len(d)} vs {len(s1)})')
    bad = sum(1 for v in d.values() for x in v if x not in s23)
    if bad:
        errors.append(f'{name}: {bad} ids not in test S2/S3')
notsub = sum(1 for k, v in m.items() if not set(v) <= set(c.get(k, [])))
if notsub:
    errors.append(f'{notsub} S1 rows have matches outside their candidates')
dup_global = sum(len(v) for v in m.values()) - len({x for v in m.values() for x in v})
n_m = sum(len(v) for v in m.values()); n_c = sum(len(v) for v in c.values())
print(f'S1 rows {len(m)}; matches {n_m} ({n_m/len(m):.2f}/S1); empty {sum(not v for v in m.values())/len(m):.3f}; '
      f'candidates {n_c} ({n_c/len(c):.1f}/S1); S2/S3 ids matched to >1 S1: {dup_global}')
print('\n'.join(errors) if errors else 'ALL CHECKS PASSED')
sys.exit(1 if errors else 0)
