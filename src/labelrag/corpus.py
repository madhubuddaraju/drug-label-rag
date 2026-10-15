"""Turn drug labels into retrievable chunks, and resolve drug names in questions.

The tokenizer and drug matching here are mirrored line for line in dashboard/rag.js; tests check they agree.
"""

import gzip
import json
import re

from .config import CHUNK_OVERLAP, CHUNK_WORDS, CHUNKS, LABELS, SECTION_TITLES

STOP = set("""a an and are as at be been but by can could do does for from had has have how i if in into is it its
may me might my no not of on or our should so such than that the their then there these they this to was we were
what when where which while who whom why will with would you your about after also any before being both each
""".split())

SALTS = set("""hydrochloride hcl sodium potassium calcium magnesium maleate besylate tartrate succinate fumarate citrate
mesylate sulfate phosphate acetate bromide chloride hyclate monohydrate dihydrate trihydrate hemifumarate dipropionate
propionate valerate furoate carbonate bisulfate hydrobromide dihydrochloride pamoate tromethamine bitartrate
medoxomil cypionate acetonide er xr extended release delayed tablets tablet capsules capsule oral injection usp
""".split())

FORMULATION = set("xl sr cr er xr la cd odt ds xt hfa ped pf ls sprinkle sprinkles".split())
BAD_ALIAS = set("""in non proprietary name release extended suspension solution oral bacteriostatic antiseptic cleanser
water dextrose injection kit pak""".split())

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")


# question wording -> label section it is probably about. Plain regex strings so the browser compiles the same patterns.
SECTION_HINTS = [
    ("pediatric", r"\b(kids?|child|children|pediatric|paediatric|teens?|teenagers?|adolescents?|infants?|bab(y|ies))\b"),
    ("geriatric", r"\b(elderly|older (adults?|people|patients)|seniors?|geriatric)\b"),
    ("pregnancy", r"\b(pregnan\w*|trimester|fetus|fetal)\b"),
    ("lactation", r"\b(breast ?feed\w*|nursing|breast ?milk|lactat\w*)\b"),
    ("overdosage", r"\b(overdos\w*|too much|too many)\b"),
    ("adverse_reactions", r"\b(side effects?|adverse)\b"),
    ("dosage", r"\b(doses?|dosing|dosage|how much|how (do|should) i take|how to take|max(imum)?)\b"),
    ("interactions", r"\b(interact\w*|together|combin\w*|mix\w*)\b|\bwith (alcohol|grapefruit|other)\b"),
    ("contraindications", r"\b(contraindicat\w*|should(n't| not) (take|use)|who (can't|cannot|should not))\b"),
    ("indications", r"\b(used for|use for|treats?|indicat\w*)\b"),
    ("boxed_warning", r"\b(boxed|black box)\b"),
]
_HINTS = [(sec, re.compile(rx, re.I)) for sec, rx in SECTION_HINTS]


def section_hints(question: str) -> list[str]:
    return [sec for sec, rx in _HINTS if rx.search(question)]


def expand_query(question: str) -> str:
    """Append the titles of the sections a question points at, so 'kids' can match 'Pediatric use'."""
    hints = section_hints(question)
    return question + "".join(f" {SECTION_TITLES[h]}" for h in hints)


def stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP and (len(t) > 1 or t.isdigit())]


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def chunk_text(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack whole sentences into ~`words`-word chunks, carrying ~`overlap` words of context into the next chunk."""
    sents = []
    for s in split_sentences(text):
        w = s.split()
        # very long "sentences" (tables flattened to text) are cut into word windows
        sents += [" ".join(w[i:i + words]) for i in range(0, len(w), words)] if len(w) > words else [s]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > words:
            chunks.append(" ".join(cur))
            keep = []
            for x in reversed(cur):
                if sum(len(k.split()) for k in keep) >= overlap:
                    break
                keep.insert(0, x)
            cur = keep if sum(len(k.split()) for k in keep) < words // 2 else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def display_name(drug: str) -> str:
    return drug[:1].upper() + drug[1:]


def header(label: dict, section: str) -> str:
    brands = ", ".join(a.title() for a in label["aliases"][:3])
    return f"{display_name(label['drug'])}{f' ({brands})' if brands else ''} | {SECTION_TITLES[section]}"


def clean_aliases(label: dict) -> list[str]:
    """Keep real brand names: drop generic variants ("metformin hydrochloride er"), strengths and parentheticals."""
    generic = set(base_name(label["drug"]).split())
    out = []
    for a in label.get("aliases", []):
        a = re.sub(r"\(.*?\)", "", a).strip(" ,-")
        toks = set(_TOKEN.findall(a))
        if toks & BAD_ALIAS or all(len(t) == 1 for t in toks):
            continue
        if not a or re.search(r"\d|,|\busp\b", a) or toks & generic or len(a) < 3 or a in out:
            continue
        out.append(a)
        words = a.split()
        while len(words) > 1 and words[-1] in FORMULATION:   # "wellbutrin xl" -> also "wellbutrin"
            words = words[:-1]
            if " ".join(words) not in out:
                out.append(" ".join(words))
    return out[:10]


def load_labels(path=LABELS) -> list[dict]:
    with gzip.open(path, "rt") as f:
        labels = [json.loads(line) for line in f]
    for lab in labels:
        lab["aliases"] = clean_aliases(lab)
    return labels


def build_chunks(labels: list[dict]) -> list[dict]:
    out = []
    for lab in labels:
        for sec, text in lab["sections"].items():
            for j, piece in enumerate(chunk_text(text)):
                out.append({"id": len(out), "drug": lab["drug"], "section": sec, "part": j,
                            "header": header(lab, sec), "text": piece})
    return out


def save_chunks(chunks, path=CHUNKS):
    with gzip.open(path, "wt") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def load_chunks(path=CHUNKS) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def base_name(name: str) -> str:
    return " ".join(t for t in name.split() if t not in SALTS)


def alias_index(labels: list[dict]) -> dict[str, list[str]]:
    """Map every way of naming a drug (generic, generic without salt, brand names) to the drugs it may mean."""
    idx: dict[str, set] = {}
    for lab in labels:
        names = {lab["drug"], base_name(lab["drug"])} | set(lab["aliases"]) | {base_name(a) for a in lab["aliases"]}
        for n in names:
            key = " ".join(_TOKEN.findall(n.lower()))
            if len(key) >= 3:
                idx.setdefault(key, set()).add(lab["drug"])
    return {k: sorted(v) for k, v in idx.items()}


def find_drugs(question: str, aliases: dict[str, list[str]], max_len: int = 5) -> list[str]:
    """Longest-match lookup of drug names in a question. Returns the drugs mentioned (may be empty)."""
    toks = _TOKEN.findall(question.lower())
    found, i = [], 0
    while i < len(toks):
        for n in range(min(max_len, len(toks) - i), 0, -1):
            key = " ".join(toks[i:i + n])
            if key in aliases:
                found += [d for d in aliases[key] if d not in found]
                i += n
                break
        else:
            i += 1
    return found
