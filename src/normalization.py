"""Policy v1: conservative one-pass dictionary substitution, Unicode L/Nd tokens."""
from collections import defaultdict
import csv
import unicodedata


def cleanup(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def canonical(text):
    return cleanup(cleanup(text).casefold())


def token_char(char):
    return unicodedata.category(char).startswith("L") or unicodedata.category(char) == "Nd"


def spans(text):
    start = None
    for i, char in enumerate(text + " "):
        if token_char(char):
            if start is None:
                start = i
        elif start is not None:
            yield start, i
            start = None


def load_dictionary(path):
    candidates = defaultdict(set)
    rows = ineligible = empty = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            key, value = canonical(row["slang"]), canonical(row["formal"])
            if not key or not all(token_char(c) for c in key):
                ineligible += 1
                continue
            if not value:
                empty += 1
                continue
            candidates[key].add(value)
    ambiguous = {k: sorted(v) for k,v in candidates.items() if len(v)>1}
    identities = {k for k,v in candidates.items() if len(v)==1 and k in v}
    mapping = {k: next(iter(v)) for k,v in candidates.items() if len(v)==1 and k not in identities}
    stats = {"rows": rows, "ineligible_key_rows": ineligible, "empty_value_rows": empty,
             "eligible_unique_keys":len(candidates), "ambiguous_keys_skipped":len(ambiguous),
             "identity_keys_skipped":len(identities), "active_mapping_keys":len(mapping)}
    return mapping, stats, ambiguous


def normalize(text, mapping):
    original = cleanup(text)
    parts, changes, pos = [], [], 0
    for start, end in spans(original):
        token = original[start:end]
        replacement = mapping.get(canonical(token))
        if replacement is None:
            continue
        parts.extend([original[pos:start],replacement])
        changes.append({"start":start,"end":end,"original":token,"replacement":replacement})
        pos = end
    parts.append(original[pos:])
    return original, "".join(parts), changes


def word_tokens(text):
    text = text.casefold()
    result, pos = [], 0
    for start,end in spans(text):
        result.extend(c for c in text[pos:start] if not c.isspace())
        result.append(text[start:end])
        pos=end
    result.extend(c for c in text[pos:] if not c.isspace())
    return result
