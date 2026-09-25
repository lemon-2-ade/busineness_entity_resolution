"""Exploratory data analysis for the Business ER challenge.

Prints the statistics that drove the design decisions.
Usage: python scripts/eda.py --data data
"""
import argparse, re, collections
import polars as pl

def rd(p):
    return pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False)

NONLATIN = re.compile(r'[^\x00-ɏ\s]')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data'); a = ap.parse_args()
    d = a.data
    s1 = rd(f'{d}/train/train_source1.tsv'); s2 = rd(f'{d}/train/train_source2.tsv'); s3 = rd(f'{d}/train/train_source3.tsv')
    gt = rd(f'{d}/train/train_ground_truth.tsv').fill_null('')
    print('rows', s1.height, s2.height, s3.height)
    for n, df in [('s1', s1), ('s2', s2), ('s3', s3)]:
        nm = df['business_name']; ad = df['business_address'].fill_null('')
        print(n, 'country', dict(df['country'].value_counts().iter_rows()),
              '| nonlatin name %.3f' % nm.str.contains(NONLATIN.pattern).mean(),
              '| nonlatin addr %.3f' % ad.str.contains(NONLATIN.pattern).mean(),
              '| empty addr %.3f' % (ad == '').mean(),
              '| domain-like name %.3f' % nm.str.contains(r'(?i)\.com|www\.').mean(),
              '| dba/aka %.3f' % nm.str.contains(r'(?i)\bdba\b|d/b/a|trading as|formerly|\baka\b').mean())
    k = gt['matched_entity_ids'].map_elements(lambda s: len(s.split(',')) if s else 0, return_dtype=pl.Int64)
    print('singleton rate', (k == 0).mean(), 'mean cluster size', k.mean())
    ex = gt.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
    print('pairs', ex.height, 'unique S2/S3 in pairs', ex['matched_entity_ids'].n_unique(), '(=> each S2/S3 record matches at most one S1)')
    print('unmatched S2/S3 records', s2.height + s3.height - ex.height)
    # country consistency within true pairs
    allr = pl.concat([s2, s3]).select('entity_id', pl.col('country').alias('c2'))
    j = ex.join(s1.select('entity_id', 'country'), left_on='source1_entity_id', right_on='entity_id').join(allr, left_on='matched_entity_ids', right_on='entity_id')
    print('pairs with same country', (j['country'] == j['c2']).mean())

if __name__ == '__main__':
    main()
