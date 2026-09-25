"""I/O helpers.  All files are TSV with no quoting (addresses contain commas/quotes)."""
import polars as pl


def read_tsv(path, **kw):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False, **kw)


def read_source(path):
    df = read_tsv(path)
    return df.with_columns(pl.col('business_name').fill_null(''), pl.col('business_address').fill_null(''),
                           pl.col('country').fill_null(''))


def read_ground_truth(path):
    """Returns (gt, pairs): gt = one row per S1 entity, pairs = exploded (s1, matched)."""
    gt = read_tsv(path).with_columns(pl.col('matched_entity_ids').fill_null(''))
    pairs = (gt.with_columns(pl.col('matched_entity_ids').str.split(','))
             .explode('matched_entity_ids')
             .filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != ''))
             .rename({'matched_entity_ids': 'matched_entity_id'}))
    return gt, pairs


def write_id_lists(path, s1_ids, lists, col):
    """Write `source1_entity_id \\t <col>` with comma-joined id lists."""
    with open(path, 'w') as f:
        f.write(f'source1_entity_id\t{col}\n')
        for s, l in zip(s1_ids, lists):
            f.write(f"{s}\t{','.join(l)}\n")
