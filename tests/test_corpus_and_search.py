import numpy as np

from labelrag import corpus
from labelrag.config import CHUNK_WORDS
from labelrag.evaluate import relevant, same_drug, score_ranking
from labelrag.search import BM25, Retriever, hamming_scores, rrf, to_binary


def test_chunks_are_bounded_and_cover_every_section(labels, chunks):
    words = [len(c["text"].split()) for c in chunks]
    assert max(words) <= CHUNK_WORDS * 1.5
    assert {(c["drug"], c["section"]) for c in chunks} == {(lab["drug"], s) for lab in labels for s in lab["sections"]}
    assert [c["id"] for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap_for_context():
    text = " ".join(f"Sentence number {i} says something about dosing and safety." for i in range(60))
    parts = corpus.chunk_text(text, words=40, overlap=10)
    assert len(parts) > 3
    assert parts[0].split(". ")[-1] in parts[1]


def test_aliases_keep_brands_and_drop_generic_variants(labels):
    by = {lab["drug"]: lab for lab in labels}
    assert "neurontin" in by["gabapentin"]["aliases"]
    assert not any("metformin" in a for a in by["metformin hydrochloride"]["aliases"])
    assert "wellbutrin" in by["bupropion hydrochloride"]["aliases"]     # derived from "wellbutrin xl"


def test_find_drugs_resolves_brand_generic_and_salt_free_names(aliases):
    assert corpus.find_drugs("Is Lipitor safe in pregnancy?", aliases) == ["atorvastatin calcium"]
    assert corpus.find_drugs("max dose of metformin?", aliases) == ["metformin hydrochloride"]
    assert set(corpus.find_drugs("ibuprofen with warfarin", aliases)) == {"ibuprofen", "warfarin sodium"}
    assert corpus.find_drugs("What is the capital of France?", aliases) == []


def test_bm25_ranks_matching_document_first():
    bm = BM25(["warfarin bleeding risk", "metformin lactic acidosis", "gabapentin dizziness somnolence"])
    s = bm.scores("lactic acidosis")
    assert int(np.argmax(s)) == 1 and s[0] == 0


def test_binary_quantisation_counts_matching_bits():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5, 64))
    bits = to_binary(emb)
    assert bits.shape == (5, 8)
    s = hamming_scores(to_binary(emb[2:3]), bits)
    assert s[2] == 64 and s.argmax() == 2


def test_rrf_rewards_agreement():
    fused = rrf([np.array([1, 2, 3]), np.array([2, 1, 4])])
    assert fused[1] == fused[2] > fused[3] == fused[4]


def test_drug_filter_keeps_results_on_the_named_drug(chunks, aliases):
    r = Retriever(chunks, aliases)
    ids = r.search("What is the usual dose of Neurontin?", method="bm25", k=5)
    assert ids and all(chunks[i]["drug"] == "gabapentin" for i in ids)
    assert chunks[ids[0]]["section"] == "dosage"


def test_metrics_and_drug_families(chunks):
    assert same_drug("sertraline", "sertraline hydrochloride")
    q = {"drug": "sertraline", "sections": ["dosage"]}
    gold = relevant(chunks, q)
    assert any(chunks[i]["drug"] == "sertraline hydrochloride" for i in gold)
    first = sorted(gold)[0]
    other = next(c["id"] for c in chunks if c["id"] not in gold)
    m = score_ranking([other, first], gold, chunks, "sertraline")
    assert m["hit@1"] == 0 and m["hit@5"] == 1 and m["mrr@10"] == 0.5


def test_section_hints_map_everyday_words_to_label_sections():
    assert corpus.section_hints("Can kids take Neurontin?") == ["pediatric"]
    assert corpus.section_hints("What happens if I take too much?") == ["overdosage"]
    assert "lactation" in corpus.section_hints("Is it safe while breastfeeding?")
    assert corpus.expand_query("Can kids take it?").endswith("Pediatric use")
