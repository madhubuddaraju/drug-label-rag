import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from labelrag import generate
from labelrag.search import BM25, doc_text, hamming_scores, to_binary
from labelrag.web import browser_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_answer_parsing_helpers():
    a = "Take 500 mg twice daily with meals [1]. The maximum is 2,550 mg per day [1][3]."
    assert generate.citations(a) == [1, 3]
    assert generate.sentences(a) == ["Take 500 mg twice daily with meals.", "The maximum is 2,550 mg per day."]
    assert generate.is_refusal(generate.REFUSAL) and not generate.is_refusal(a)
    assert generate.key_facts(a, [["2,550", "2550"], ["insulin"]]) == 0.5
    assert generate.key_facts(a, []) is None


def test_summary_counts_refusals_and_citations():
    rows = [
        {"source": "natural", "refused": False, "retrieved_gold": True, "citations": [1], "cited_right_drug": [True],
         "cited_right_section": [True], "sentences": 2, "supported": 2, "key_facts": 1.0, "closed_book_key_facts": 0.0, "seconds": 3},
        {"source": "natural", "refused": True, "retrieved_gold": True, "citations": [], "cited_right_drug": [],
         "cited_right_section": [], "key_facts": 0.0, "closed_book_key_facts": 1.0, "seconds": 2},
        {"source": "unanswerable", "refused": True, "citations": [], "seconds": 1},
    ]
    s = generate.summarise(rows)
    assert s["answer_rate"] == 0.5 and s["correct_refusal_rate"] == 1.0
    assert s["faithfulness"] == 1.0 and s["citation_right_section"] == 1.0
    assert s["false_refusal_when_gold_retrieved"] == 0.5


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_engine_matches_python(tmp_path, labels, chunks, aliases):
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(len(chunks), 384)).astype(np.float32)
    payload = browser_corpus(labels, chunks, emb)
    queries = ["Can I drink alcohol with Flagyl?", "max dose of metformin", "Xanax overdose flumazenil",
               "is Lipitor safe in pregnancy", "side effects of neurontin dizziness", "Can kids take Neurontin?",
               "Is it OK to breastfeed on Zoloft? Who shouldn't take it?"]
    qvecs = rng.normal(size=(len(queries), 384))
    (tmp_path / "corpus.json").write_text(json.dumps(payload))
    (tmp_path / "q.json").write_text(json.dumps({"queries": queries, "qvecs": qvecs.tolist()}))
    script = f"""
const {{ makeEngine }} = require({json.dumps(str(ROOT / "dashboard" / "rag.js"))});
const fs = require("fs");
global.atob = (s) => Buffer.from(s, "base64").toString("binary");
const D = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "corpus.json"))}));
const Q = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "q.json"))}));
const E = makeEngine(D);
const out = Q.queries.map((q, i) => ({{ bm25: Array.from(E.bm25(E.expandQuery(q))), drugs: E.findDrugs(q), expanded: E.expandQuery(q),
  bin: Array.from(E.binaryScores(Q.qvecs[i])) }}));
process.stdout.write(JSON.stringify(out));
"""
    res = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    bm = BM25([doc_text(c) for c in chunks])
    from labelrag.corpus import expand_query, find_drugs
    for q, v, r in zip(queries, qvecs, res, strict=True):
        assert expand_query(q) == r["expanded"]
        assert np.allclose(bm.scores(expand_query(q)), r["bm25"], atol=1e-9)
        assert find_drugs(q, aliases) == r["drugs"]
        assert np.array_equal(hamming_scores(to_binary(v[None]), to_binary(emb)), np.array(r["bin"]))
