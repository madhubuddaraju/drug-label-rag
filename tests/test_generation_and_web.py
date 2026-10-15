import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from labelrag import grade
from labelrag.search import BM25, doc_text, hamming_scores, to_binary
from labelrag.web import browser_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_answer_parsing_helpers():
    a = "Take 500 mg twice daily with meals [1]. The maximum is 2,550 mg per day. [1][3]"
    assert grade.citations(a) == [1, 3]
    assert grade.sentences(a) == ["Take 500 mg twice daily with meals.", "The maximum is 2,550 mg per day."]
    assert [r for r, _ in grade.split_answer(a)][1] == "The maximum is 2,550 mg per day [1][3]."
    assert grade.is_refusal(grade.REFUSAL) and not grade.is_refusal(a)
    assert grade.key_facts(a, [["2,550", "2550"], ["insulin"]]) == 0.5
    assert grade.key_facts(a, []) is None
    b = "The maximum is 2550 mg [1, 3]. Take it with food [2], [3]."
    assert grade.citations(b) == [1, 2, 3]
    assert grade.split_answer(b) == [("The maximum is 2550 mg [1, 3].", "The maximum is 2550 mg."),
                                     ("Take it with food [2], [3].", "Take it with food.")]


class FakeJudge:
    """Entails a claim when every word of it appears in the passages."""
    name = "fake"

    def support(self, claims, passages):
        words = set(" ".join(" ".join(p) if isinstance(p, tuple) else p for p in passages).lower().replace(".", "").split())
        return np.array([float(set(c.lower().rstrip(".").split()) <= words) for c in claims])


def test_guardrail_drops_unsupported_claims_and_refuses_when_nothing_is_left():
    passages = ["the maximum dose is 2550 mg per day. take with meals."]
    row = {"source": "natural", "answer": "The maximum dose is 2550 mg per day [1]. It is safe in pregnancy [1].",
           "must": [["2550"]], "seconds": 1}
    g = generate_grade(row, passages)
    assert g["claims"] == 2 and g["supported"] == 1
    assert g["guarded_answer"] == "The maximum dose is 2550 mg per day [1]." and not g["guarded_refused"]
    assert g["key_facts"] == 1.0 and g["guarded_key_facts"] == 1.0
    g = generate_grade({**row, "answer": "It is safe in pregnancy [1]."}, passages)
    assert g["guarded_refused"] and g["guarded_answer"] == grade.REFUSAL


def test_passage_headers_reach_the_judge():
    g = grade.grade_row({"source": "natural", "answer": "Lisinopril starts at 10 mg daily [1].", "seconds": 1},
                        [("Lisinopril", "starts at 10 mg daily.")], FakeJudge())
    assert g["supported"] == 1


def test_drug_gate_refuses_when_no_known_drug_is_named():
    row = {"source": "unanswerable", "answer": "The maximum dose is 2550 mg per day [1].", "seconds": 1}
    g = grade.grade_row(row, ["the maximum dose is 2550 mg per day."], FakeJudge(), drug_found=False)
    assert g["supported"] == 1 and g["guarded_refused"] and g["guarded_answer"] == grade.NO_DRUG
    assert g["guarded_by"] == "drug gate"
    # stored support scores can be reused without the judge
    again = grade.grade_row(g, ["unused"], None, drug_found=True)
    assert again["guarded_answer"] == "The maximum dose is 2550 mg per day [1]." and not again["guarded_refused"]


def generate_grade(row, passages):
    return grade.grade_row(row, passages, FakeJudge())


def test_summary_counts_refusals_and_citations():
    passages = ["take 500 mg twice daily with meals."]
    rows = [
        grade.grade_row({"source": "natural", "answer": "Take 500 mg twice daily with meals [1].", "retrieved_gold": True,
                         "citations": [1], "cited_right_drug": [True], "cited_right_section": [True], "must": [["500 mg"]],
                         "closed_book_key_facts": 0.0, "seconds": 3}, passages, FakeJudge()),
        grade.grade_row({"source": "natural", "answer": grade.REFUSAL, "retrieved_gold": True, "citations": [],
                         "must": [["500 mg"]], "closed_book_key_facts": 1.0, "seconds": 2}, passages, FakeJudge()),
        grade.grade_row({"source": "unanswerable", "answer": grade.REFUSAL, "citations": [], "seconds": 1}, passages, FakeJudge()),
    ]
    s = grade.summarise(rows)
    assert s["raw"]["answer_rate"] == 0.5 and s["raw"]["correct_refusal_rate"] == 1.0
    assert s["raw"]["faithfulness"] == 1.0 and s["raw"]["citation_right_section"] == 1.0
    assert s["raw"]["false_refusal_when_gold_retrieved"] == 0.5
    assert s["guarded"]["claims_removed"] == 0.0 and s["key_fact_recall_closed_book"] == 0.5


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
