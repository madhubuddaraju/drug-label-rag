// Browser retrieval engine. Mirrors src/labelrag/corpus.py and search.py:
// same tokenizer and stemmer, same BM25 (k1=1.2, b=0.75), same drug-name lookup, binary embeddings
// compared by matching bits, and reciprocal rank fusion. tests/test_web_parity.py checks they agree.
function makeEngine(D) {
  const STOP = new Set(D.stop);
  const stem = (t) => (t.length > 4 && t.endsWith("ies") ? t.slice(0, -3) + "y"
    : t.length > 3 && t.endsWith("s") && !/(ss|us|is)$/.test(t) ? t.slice(0, -1) : t);
  const words = (s) => s.toLowerCase().match(/[a-z0-9]+/g) || [];
  const tokenize = (s) => words(s).filter((t) => !STOP.has(t) && (t.length > 1 || /^\d+$/.test(t))).map(stem);

  // ---- BM25 index, built once from the chunk text ----
  const k1 = 1.2, b = 0.75, N = D.chunks.length;
  const postings = new Map();   // term -> [[doc, weight], ...]
  const lens = new Float64Array(N);
  const tfs = D.chunks.map((c, i) => {
    const toks = tokenize(c.header + ". " + c.text);
    lens[i] = toks.length;
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    return tf;
  });
  const avg = lens.reduce((a, v) => a + v, 0) / (N || 1);
  tfs.forEach((tf, i) => {
    const norm = k1 * (1 - b + b * lens[i] / avg);
    for (const [t, c] of tf) {
      if (!postings.has(t)) postings.set(t, []);
      postings.get(t).push([i, c * (k1 + 1) / (c + norm)]);
    }
  });
  const idf = new Map();
  for (const [t, p] of postings) idf.set(t, Math.log(1 + (N - p.length + 0.5) / (p.length + 0.5)));

  function bm25(query) {
    const s = new Float64Array(N);
    for (const t of new Set(tokenize(query))) {
      const p = postings.get(t);
      if (!p) continue;
      const w = idf.get(t);
      for (const [i, v] of p) s[i] += w * v;
    }
    return s;
  }

  // ---- section hints: "kids" -> "Pediatric use" (same regexes as corpus.SECTION_HINTS) ----
  const HINTS = (D.hints || []).map(([sec, rx]) => [sec, new RegExp(rx, "i")]);
  const sectionHints = (q) => HINTS.filter(([, rx]) => rx.test(q)).map(([sec]) => sec);
  const expandQuery = (q) => q + sectionHints(q).map((s) => " " + D.sections[s]).join("");

  // ---- drug names ----
  function findDrugs(q, maxLen = 5) {
    const toks = words(q), found = [];
    let i = 0;
    while (i < toks.length) {
      let matched = false;
      for (let n = Math.min(maxLen, toks.length - i); n > 0; n--) {
        const key = toks.slice(i, i + n).join(" ");
        if (D.aliases[key]) { for (const d of D.aliases[key]) if (!found.includes(d)) found.push(d); i += n; matched = true; break; }
      }
      if (!matched) i++;
    }
    return found;
  }

  // ---- binary embeddings (48 bytes per chunk) ----
  const bits = D.bits ? Uint8Array.from(atob(D.bits), (ch) => ch.charCodeAt(0)) : null;
  const BYTES = D.dim / 8;
  const POP = new Uint8Array(256).map((_, i) => { let c = 0, x = i; while (x) { c += x & 1; x >>= 1; } return c; });
  function binaryScores(qvec) {
    const q = new Uint8Array(BYTES);
    for (let j = 0; j < D.dim; j++) if (qvec[j] > 0) q[j >> 3] |= 128 >> (j & 7);   // same bit order as numpy.packbits
    const s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let diff = 0;
      for (let j = 0; j < BYTES; j++) diff += POP[bits[i * BYTES + j] ^ q[j]];
      s[i] = D.dim - diff;
    }
    return s;
  }

  function top(scores, k, allowed) {
    const idx = [];
    for (let i = 0; i < N; i++) if ((!allowed || allowed[i]) && Number.isFinite(scores[i])) idx.push(i);
    idx.sort((a, c) => scores[c] - scores[a] || a - c);
    return idx.slice(0, k);
  }
  function rrf(lists, k = 60, depth = 100) {
    const out = new Map();
    for (const r of lists) r.slice(0, depth).forEach((i, pos) => out.set(i, (out.get(i) || 0) + 1 / (k + pos + 1)));
    return [...out.entries()].sort((a, c) => c[1] - a[1] || a[0] - c[0]).map((e) => e[0]);
  }

  function search(question, { k = 5, qvec = null, drugFilter = true, hints = true } = {}) {
    const drugs = drugFilter ? findDrugs(question) : [];
    const query = hints ? expandQuery(question) : question;
    const allowed = drugs.length ? D.chunks.map((c) => drugs.includes(c.drug)) : null;
    const lex = bm25(query);
    const lexTop = top(lex, 100, allowed);
    const ids = qvec && bits ? rrf([lexTop, top(binaryScores(qvec), 100, allowed)]).slice(0, k) : lexTop.slice(0, k);
    return { ids, drugs, hints: hints ? sectionHints(question) : [], lexBest: lexTop.length ? lex[lexTop[0]] : 0, mode: qvec && bits ? "hybrid" : "keyword" };
  }

  // best supporting sentence in a passage: most query terms covered
  function bestSentence(query, text) {
    const q = new Set(tokenize(query));
    const sents = text.split(/(?<=[.;:])\s+(?=[A-Z0-9(])/);
    let best = 0, bestScore = -1;
    sents.forEach((s, i) => { const sc = new Set(tokenize(s).filter((t) => q.has(t))).size; if (sc > bestScore) { best = i; bestScore = sc; } });
    return { sentences: sents, best, terms: q };
  }

  return { tokenize, bm25, findDrugs, expandQuery, sectionHints, binaryScores, search, bestSentence, top, rrf };
}
if (typeof module !== "undefined") module.exports = { makeEngine };
