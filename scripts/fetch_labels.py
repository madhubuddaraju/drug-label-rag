"""Download openFDA drug labels and keep the latest label for the most widely marketed prescription generics.

Runs in GitHub Actions (the full download is about 1.5 GB). Output: data/labels.jsonl.gz, one record per drug:
{"drug", "aliases", "set_id", "effective_time", "n_labels", "manufacturer", "route", "sections": {key: text}}
"""

import argparse
import collections
import gzip
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labelrag.config import LABELS, LABELS_META, N_DRUGS, SECTIONS  # noqa: E402

INDEX = "https://api.fda.gov/download.json"


def get(url, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"retry {i + 1} for {url}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise SystemExit(f"failed to download {url}")


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    # most sections start with their own heading, e.g. "2 DOSAGE AND ADMINISTRATION 2.1 Adults ..."
    return re.sub(r"^\d{0,2}\s*[A-Z][A-Z &/,\-]{4,60}?(?=\s+(?:\d{1,2}\.\d|[A-Z][a-z]))", "", s).strip()


def norm_name(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", type=int, default=N_DRUGS)
    ap.add_argument("--max-parts", type=int, default=0, help="only for quick tests")
    a = ap.parse_args()

    parts = json.loads(get(INDEX))["results"]["drug"]["label"]["partitions"]
    if a.max_parts:
        parts = parts[: a.max_parts]
    print(f"{len(parts)} partitions", flush=True)

    counts = collections.Counter()      # generic -> distinct labels (proxy for how widely a drug is marketed)
    brands = collections.defaultdict(collections.Counter)
    best = {}                           # generic -> best record so far
    seen = 0
    digest = hashlib.sha256()
    for p in parts:
        blob = get(p["file"])
        digest.update(blob)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            rows = json.load(z.open(z.namelist()[0]))["results"]
        del blob
        for r in rows:
            seen += 1
            o = r.get("openfda") or {}
            if "HUMAN PRESCRIPTION DRUG" not in (o.get("product_type") or []):
                continue
            names = o.get("generic_name") or []
            if len(names) != 1 or "dosage_and_administration" not in r:
                continue
            g = norm_name(names[0])
            if len(g) > 60:
                continue
            counts[g] += 1
            for b in o.get("brand_name") or []:
                b = norm_name(b)
                if b != g and len(b) <= 40:
                    brands[g][b] += 1
            secs = {}
            for field, (key, _) in SECTIONS.items():
                if field in r and key not in secs:
                    text = clean(" ".join(r[field]))
                    if len(text.split()) >= 5:
                        secs[key] = text
            # prefer the newest PLR-format label with the most sections
            score = (len(secs), r.get("effective_time", ""))
            if g not in best or score > best[g]["_score"]:
                best[g] = {"drug": g, "set_id": r.get("set_id"), "effective_time": r.get("effective_time"),
                           "manufacturer": (o.get("manufacturer_name") or [""])[0], "route": o.get("route") or [],
                           "sections": secs, "_score": score}
        print(f"{p['file'].rsplit('/', 1)[-1]}: {seen:,} labels read, {len(counts):,} Rx generics", flush=True)
        del rows

    top = [g for g, _ in counts.most_common() if len(best[g]["sections"]) >= 5][: a.drugs]
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(LABELS, "wt") as f:
        for g in sorted(top):
            rec = best[g]
            rec.pop("_score")
            rec["n_labels"] = counts[g]
            rec["aliases"] = [b for b, _ in brands[g].most_common(6)]
            f.write(json.dumps(rec) + "\n")
    meta = {"source": INDEX, "partitions": len(parts), "labels_read": seen, "rx_generics": len(counts),
            "drugs_kept": len(top), "sha256_of_downloads": digest.hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    LABELS_META.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
