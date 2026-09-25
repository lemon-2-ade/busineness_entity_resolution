"""Country-agnostic text normalisation for business names and addresses.

Design notes:
  * Every rule here is language/format knowledge (abbreviations, legal forms,
    honorifics, alias markers) or is *learned from the training pairs*
    (Indic-script token dictionary, see `learn_translit_dict`).  No external
    data source is consulted.
  * Nothing is keyed on the country label, so an unseen country (France in the
    test set) passes through the same code path.  French legal forms / street
    types are included in the generic dictionaries simply as more vocabulary.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from anyascii import anyascii

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
# canonical legal-form tokens (value) for surface forms (key)
LEGAL = {
    'ltd': 'ltd', 'limited': 'ltd', 'ltda': 'ltd',
    'pvt': 'pvt', 'private': 'pvt', 'pte': 'pvt',
    'inc': 'inc', 'incorporated': 'inc',
    'corp': 'corp', 'corporation': 'corp',
    'co': 'co', 'company': 'co', 'cie': 'co',
    'llc': 'llc', 'llp': 'llp', 'lp': 'lp', 'pllc': 'pllc', 'pc': 'pc', 'plc': 'plc',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sa': 'sa',
    'sci': 'sci', 'snc': 'snc', 'scop': 'scop', 'gmbh': 'gmbh',
}
# dotted / multi-part legal forms collapsed before tokenisation
LEGAL_PATTERNS = [
    (r'\bp\.?\s?l\.?\s?l\.?\s?c\b\.?', ' pllc '),
    (r'\bl\.?\s?l\.?\s?c\b\.?', ' llc '),
    (r'\bl\.?\s?l\.?\s?p\b\.?', ' llp '),
    (r'\bl\.\s?p\b\.?', ' lp '),
    (r'\bp\.\s?c\b\.?', ' pc '),
    (r'\bs\.\s?a\.\s?r\.\s?l\b\.?', ' sarl '),
    (r'\bs\.\s?a\.\s?s\.\s?u\b\.?', ' sasu '),
    (r'\bs\.\s?a\.\s?s\b\.?', ' sas '),
    (r'\be\.\s?u\.\s?r\.\s?l\b\.?', ' eurl '),
    (r'\bpvt\.?\s?ltd\b\.?', ' pvt ltd '),
    (r'\bprivate[\s\-]+limited\b', ' pvt ltd '),
    (r'\bpra\.?\s?li\b\.?', ' pvt ltd '),
]
HONORIFICS = {'mr', 'mrs', 'ms', 'miss', 'dr', 'shri', 'sri', 'shree', 'smt', 'm/s', 'ms/', 'messrs',
              'the', 'm', 'mme', 'mlle', 'sh', 'kumari', 'km'}
# generic filler words that the vendors inject; kept in the full string but
# removed from the "core" name
FILLER = {'center', 'centre', 'services', 'service', 'partners', 'group', 'and', 'of', 'de', 'des', 'du',
          'la', 'le', 'les', 'et', 'enterprises', 'enterprise', 'holdings', 'solutions', 'international'}
# alias markers: "X dba Y", "Y formerly X", ...
ALIAS_RE = re.compile(
    r'\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|trading as|t\s*/\s*a|a\s*/\s*k\s*/\s*a|aka|'
    r'also known as|known as|f\s*/\s*k\s*/\s*a|fka|formerly known as|formerly|nee|née)\b:?', re.I)

ADDR_ABBR = {
    'road': 'rd', 'roda': 'rd', 'street': 'st', 'str': 'st', 'saint': 'st', 'ste': 'st', 'avenue': 'ave', 'av': 'ave',
    'drive': 'dr', 'drv': 'dr', 'court': 'ct', 'place': 'pl', 'lane': 'ln', 'boulevard': 'blvd', 'bd': 'blvd',
    'bvd': 'blvd', 'parkway': 'pkwy', 'highway': 'hwy', 'circle': 'cir', 'trail': 'trl', 'terrace': 'ter',
    'square': 'sq', 'mount': 'mt', 'fort': 'ft', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw', 'point': 'pt', 'expressway': 'expy',
    'crossing': 'xing', 'heights': 'hts', 'junction': 'jct', 'center': 'ctr', 'centre': 'ctr', 'plaza': 'plz',
    'township': 'twp', 'apartments': 'apts', 'building': 'bldg', 'bldg': 'bldg', 'floor': 'fl', 'flr': 'fl',
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th', 'ground': 'gnd',
    'nagar': 'ngr', 'marg': 'mg', 'colony': 'col', 'sector': 'sec', 'extension': 'extn', 'ext': 'extn',
    'industrial': 'ind', 'area': 'ar', 'village': 'vill', 'post': 'po', 'district': 'dist', 'opposite': 'opp',
    'near': 'nr', 'behind': 'bh', 'cross': 'x', 'main': 'mn',
    # French street types
    'rue': 'rue', 'r': 'rue', 'allee': 'all', 'chemin': 'ch', 'impasse': 'imp', 'route': 'rte',
    'avenu': 'ave', 'faubourg': 'fbg', 'quai': 'qu', 'cours': 'crs',
    # misc
    'bombay': 'mumbai', 'keralam': 'kerala', 'orissa': 'odisha', 'bangalore': 'bengaluru', 'gurgaon': 'gurugram',
}
# address tokens carrying no identity information
ADDR_STOP = {'no', 'door', 'h', 'hn', 'hno', 'house', 'unit', 'apt', 'apartment', 'suite', 'flat', 'plot', 'null',
             'pmb', 'box', 'po', 'shop', 'office', 'room', 'rm', 'survey', 's', 'ward', 'khasra', 'kh', 'dno',
             'd', 'bis', 'n', 'nr', 'the', 'of', 'de', 'du', 'des', 'la', 'le', 'les', 'region', 'city', 'dist',
             'county', 'cdp', 'twp', 'at', 'and', 'et'}

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co',
    'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
    'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx',
    'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc',
}
IN_STATES = {
    'maharashtra': 'mh', 'delhi': 'dl', 'uttar pradesh': 'up', 'karnataka': 'ka', 'gujarat': 'gj',
    'west bengal': 'wb', 'telangana': 'tg', 'tamil nadu': 'tn', 'haryana': 'hr', 'rajasthan': 'rj',
    'kerala': 'kl', 'madhya pradesh': 'mp', 'andhra pradesh': 'ap', 'bihar': 'br', 'punjab': 'pb',
    'odisha': 'od', 'orissa': 'od', 'assam': 'as', 'jharkhand': 'jh', 'chhattisgarh': 'cg', 'goa': 'ga',
    'uttarakhand': 'uk', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'chandigarh': 'ch',
    'puducherry': 'py', 'pondicherry': 'py',
}
# India region codes that collide with US codes are kept distinct by prefixing
STATE_CANON = {}
for k, v in US_STATES.items():
    STATE_CANON[k] = 'state_' + v
for k, v in IN_STATES.items():
    STATE_CANON[k] = 'state_' + v
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())

INDIC_RE = re.compile(r'[ऀ-෿]')
_PUNCT_RE = re.compile(r"[^\w\s/\-]")
_WS_RE = re.compile(r'\s+')
_ID_RE = re.compile(r'\(id:[^)]*\)?|\b\d{7,}\b|\s-\s\d{6,}', re.I)   # "(ID: 1234..)" / phone-like numbers
_DOMAIN_RE = re.compile(r'^(?:https?://)?(?:www\.)?#?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|biz|info|io|us)\b')
_NUM_RE = re.compile(r'\d+')


# --------------------------------------------------------------------------- #
# Learned transliteration dictionary (native-script token -> english token)
# --------------------------------------------------------------------------- #
def learn_translit_dict(native_names, english_names, min_count=2):
    """Align native-script names with the Source-1 English name of the same
    entity, token by token (the vendors transliterate word by word, so
    token counts agree in >99.9% of cases).  Returns {native_token: english}."""
    cnt = defaultdict(Counter)
    for a, b in zip(native_names, english_names):
        ta = a.split()
        tb = _PUNCT_RE.sub(' ', b.lower()).split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            if INDIC_RE.search(x):
                cnt[x][y] += 1
    out = {}
    for k, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_count:
            out[k] = y
    return out


class Normalizer:
    def __init__(self, translit: dict | None = None, seg_dict: dict | None = None):
        self.translit = translit or {}
        self.seg_dict = seg_dict or {}

    # ---------------- generic ----------------
    def to_ascii(self, s: str) -> str:
        if not s:
            return ''
        if INDIC_RE.search(s):
            s = ' '.join(self.translit.get(t, t) for t in s.split())
        s = unicodedata.normalize('NFKC', s).replace('\u00b0', ' ').replace('\u00ba', ' ')
        return anyascii(s).lower()

    # ---------------- names ----------------
    def name(self, raw: str):
        """Returns (norm, core, alt, compact, legal_tokens_str, flags)."""
        s = self.to_ascii(raw)
        flags = 0
        s = _ID_RE.sub(' ', s)
        m = _DOMAIN_RE.match(s.strip())
        if m:                       # "healthoncology.com" -> "healthoncology"
            s = m.group(1).replace('-', ' ')
            flags |= 1
        s = s.replace('&', ' and ').replace('+', ' and ')
        for p, r in LEGAL_PATTERNS:
            s = re.sub(p, r, s)
        # alias split
        alt = ''
        parts = ALIAS_RE.split(s)
        if len(parts) > 1:
            flags |= 2
            parts = [p for p in parts if p.strip(' :-')]
            if len(parts) >= 2:
                s, alt = parts[0], ' '.join(parts[1:])
            elif parts:
                s = parts[0]
        norm_toks = self._name_tokens(s)
        alt_toks = self._name_tokens(alt) if alt else []
        legal = sorted({LEGAL[t] for t in norm_toks + alt_toks if t in LEGAL})
        core = [t for t in norm_toks if t not in LEGAL and t not in FILLER] or [t for t in norm_toks if t not in LEGAL]
        alt_core = [t for t in alt_toks if t not in LEGAL and t not in FILLER]
        norm = ' '.join(LEGAL.get(t, t) for t in norm_toks)
        return (norm, ' '.join(core), ' '.join(alt_core), ''.join(core), ' '.join(legal), flags)

    @staticmethod
    def _name_tokens(s):
        s = _PUNCT_RE.sub(' ', s).replace('/', ' ').replace('-', ' ')
        toks = [t for t in s.split() if t not in HONORIFICS]
        return toks

    # ---------------- addresses ----------------
    def address(self, raw: str):
        """Returns (norm, core, nums, state).  `core` drops stop/state tokens,
        `nums` is the set of numeric tokens (leading zeros stripped)."""
        if not raw:
            return ('', '', '', '')
        segs = []
        for seg in raw.split(','):
            seg = seg.strip()
            if seg and INDIC_RE.search(seg) and seg in self.seg_dict:
                seg = self.seg_dict[seg]
            seg = self.to_ascii(seg).strip(' .')
            if seg in STATE_CANON:
                seg = STATE_CANON[seg]
            elif seg in STATE_CODES:
                seg = 'state_' + seg
            segs.append(seg)
        s = ' , '.join(segs).replace('&', ' and ')
        s = re.sub(r'[#.,;:()\[\]\'"]', ' ', s)
        toks = []
        nums = []
        for t in s.split():
            t = ADDR_ABBR.get(t, t)
            toks.append(t)
            for n in _NUM_RE.findall(t):
                n = n.lstrip('0') or '0'
                nums.append(n)
        state = ' '.join(sorted({t for t in toks if t.startswith('state_')}))
        core = []
        for t in toks:
            if t.startswith('state_') or t in ADDR_STOP:
                continue
            for q in t.replace('/', ' ').replace('-', ' ').split():
                if q.isdigit():
                    q = q.lstrip('0') or '0'
                core.append(q)
        seen = set(); unums = []
        for n in nums:
            if n not in seen:
                seen.add(n); unums.append(n)
        return (' '.join(toks), ' '.join(core), ' '.join(unums), state)


def learn_segment_dict(native_addrs, english_addrs, min_count=5):
    """Native-script address segments (almost always the state name) are mapped
    to the English segment of the paired Source-1 address that is missing from
    the transliterated address.  Learned purely from training pairs."""
    cnt = defaultdict(Counter)
    for a, b in zip(native_addrs, english_addrs):
        if not a or not b:
            continue
        nat = [x.strip() for x in a.split(',') if INDIC_RE.search(x)]
        if len(nat) != 1:
            continue
        lat = {x.strip().lower() for x in a.split(',') if not INDIC_RE.search(x)}
        cand = [x.strip() for x in b.split(',') if x.strip().lower() not in lat]
        if len(cand) == 1:
            cnt[nat[0]][cand[0]] += 1
    out = {}
    for k, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) > 0.5:
            out[k] = y
    return out


def build_dicts_from_train(s1, s23, gt_pairs):
    """s1/s23: polars frames (entity_id, business_name, business_address);
    gt_pairs: frame (source1_entity_id, matched_entity_id).
    Learns the native-script name-token and address-segment dictionaries."""
    import polars as pl
    nat = s23.filter(pl.col('business_name').str.contains(INDIC_RE.pattern) |
                     pl.col('business_address').fill_null('').str.contains(INDIC_RE.pattern))
    j = (nat.join(gt_pairs, left_on='entity_id', right_on='matched_entity_id')
         .join(s1.select('entity_id', pl.col('business_name').alias('n1'), pl.col('business_address').alias('a1')),
               left_on='source1_entity_id', right_on='entity_id'))
    translit = learn_translit_dict(j['business_name'].to_list(), j['n1'].to_list())
    seg = learn_segment_dict(j['business_address'].fill_null('').to_list(), j['a1'].fill_null('').to_list())
    return translit, seg
