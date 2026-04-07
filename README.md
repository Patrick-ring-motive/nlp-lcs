# nlp-lcs

LCS-based sequence matching utilities for NLP pipelines.

Standard edit-distance metrics (Levenshtein) measure transformation cost, which suits typo correction but penalizes legitimate paraphrases. LCS measures **shared skeleton** — the longest subsequence preserved across both inputs — which is more semantically load-bearing for natural language comparison.

The variants compose into a recursive hierarchy:

```
character-level → word-level (LCWS) → sentence-level → …
```

with per-comparator LRU memoization making the recursive cases tractable via high cache hit rates on repeated vocabulary across a corpus.

---

## Install

```bash
npm install nlp-lcs
```

```js
const { seq, word, sentence } = require('nlp-lcs');
```

---

## API overview

Functions are organized into three namespaces based on the sequence granularity they operate on.

### `seq` — general / arbitrary-element sequences

| Function | Description |
|---|---|
| `seq.lcs(a, b, compare?, minScore?)` | Core LCS score between any two iterables. Pluggable comparator enables the recursive hierarchy. Optional `minScore` (number or function) enables early termination when a threshold can't be reached. |
| `seq.subsequence(a, b, compare?)` | Like `seq.lcs` but backtracks the DP table to return `{ score, subsequence }` with the actual matched elements. |
| `seq.match(a, b, lcs?)` | `true` if LCS score ≥ 80 % of the **longer** sequence's length. |
| `seq.has(a, b, lcs?)` | `true` if LCS score ≥ 80 % of the **shorter** sequence's length (containment check). |
| `seq.bestMatch(query, candidates, lcs?, matcher?)` | Find the highest-scoring candidate; returns `{ value, match, score }`. |
| `seq.bestHas(query, candidates, lcs?)` | Same as `bestMatch` but using the containment threshold. |
| `seq.scoreMatch(a, b, score)` | Raw threshold test against max length. |
| `seq.scoreHas(a, b, score)` | Raw threshold test against min length. |
| `seq.weighted(a, b, lcs?)` | LCS score × (minLen / maxLen) — penalizes length disparity. |
| `seq.bestWeighted(query, candidates)` | Best weighted match from a list. |
| `seq.context(a, b, lcs?)` | LCS score + (maxLen / minLen) — rewards longer containing sequences. |
| `seq.bestContext(query, candidates)` | Best context match from a list. |
| `seq.sorted(a, b, lcs?)` | LCS on sorted copies — order-insensitive comparison. |

### `word` — character-level (single-word) comparisons

All functions operate on normalized (NFD-decomposed, lowercased) strings.

| Function | Description |
|---|---|
| `word.norm(x)` | Normalize a value to a trimmed, NFD, lowercased string. |
| `word.lcs(a, b)` | Normalized character-level LCS score. |
| `word.match(a, b)` | Fuzzy word match (80 % character-level threshold). |
| `word.bestMatch(query, candidates)` | Best fuzzy word from a list. |
| `word.weighted(a, b)` | Weighted character-level score. |
| `word.bestWeighted(query, candidates)` | Best weighted word from a list. |
| `word.context(a, b)` | Context-rewarding character-level score. |
| `word.bestContext(query, candidates)` | Best context word from a list. |
| `word.sorted(a, b)` | Sorted word-level LCS. |

### `sentence` — tokenized-sentence (word-sequence) comparisons

Sentence-level functions use `word.match` as the element comparator, so structurally similar sentences align even when individual words differ slightly.

| Function | Description |
|---|---|
| `sentence.lcs(words1, words2)` | Word-level fuzzy LCS score between two token arrays. |
| `sentence.match(words1, words2)` | `true` if the sentences meet the 80 % word-structure threshold. |
| `sentence.bestMatch(query, candidates)` | Best matching sentence from a list. |
| `sentence.tokenize(str)` | Split on whitespace. |
| `sentence.toSentences(str)` | Split on `.!?` punctuation. |
| `sentence.toPhrases(str)` | Split on `.!?,;` punctuation. |
| `sentence.dedupeSegments(segments, opts?)` | Remove fuzzy-duplicate text segments. |
| `sentence.dedupeSentences(text, opts?)` | Split text into sentences and dedupe. |
| `sentence.dedupePhrases(text, opts?)` | Split text into phrases and dedupe. |

### Utility exports

These are also exported at the top level for advanced use:

- `defaultCompare` — strict equality comparator
- `createLruCache(limit?)` — standalone LRU cache factory
- `isString(x)` — string type check
- `stringify(x)` — safe JSON/string coercion
- `makeMemoKey(a, b)` — order-independent cache key builder
- `selectArrayType(maxValue)` — pick the narrowest typed array for a DP table

---

## Use cases

### 1. Deduplicating LLM sentence output

LLMs frequently produce near-duplicate sentences — restated conclusions, repeated caveats, or paraphrased bullet points. Exact dedup misses these; embedding cosine similarity is expensive. `sentence.dedupeSentences` catches fuzzy duplicates in a single pass:

```js
const { sentence } = require('nlp-lcs');

const raw = `
  The model performed well on the benchmark. Performance on the benchmark was strong.
  We recommend further evaluation. Further evaluation is recommended.
`;

const deduped = sentence.dedupeSentences(raw);
// → ['The model performed well on the benchmark',
//    'We recommend further evaluation']
```

The fuzzy word-level comparison means "performed well" and "performance was strong" are recognized as structurally overlapping, and the longer phrasing is kept by default.

### 2. Fallback matching for n-gram language models

When an n-gram model misses an exact match (unseen n-gram, novel inflection, slight misspelling), you need the closest known n-gram. `word.bestMatch` finds it without an embedding lookup:

```js
const { word } = require('nlp-lcs');

const vocab = ['running', 'jumping', 'swimming', 'walking', 'sitting'];

const { value, match } = word.bestMatch('runing', vocab);
// → { value: 'running', match: true, score: 6 }
```

Because comparison is character-level LCS rather than edit distance, legitimate morphological variation ("run" → "running") scores high while unrelated words of similar length don't.

### 3. Reranking vector database results

Vector search returns candidates ranked by embedding similarity, but embeddings can over-weight semantic relatedness at the expense of structural fidelity. `seq.weighted` and `seq.context` provide complementary reranking signals:

```js
const { seq, sentence } = require('nlp-lcs');

const query = sentence.tokenize('how to reset password');
const vectorResults = [
  'how to change your password',     // semantically close
  'how to reset password on mobile',  // structurally close + more context
  'password security best practices', // topically related but different structure
];

const reranked = vectorResults
  .map(r => {
    const tokens = sentence.tokenize(r);
    return { text: r, score: seq.context(query, tokens, sentence.lcs) };
  })
  .sort((a, b) => b.score - a.score);

// "how to reset password on mobile" ranks first — it preserves the query
// skeleton AND carries more surrounding context.
```

`seq.weighted` is better when you want tight paraphrase matches (penalizes length disparity); `seq.context` is better when you want the most informative containing passage.

### 4. Finding the relevant paragraph in an article

Given a query phrase and a long article, `seq.bestContext` scans paragraphs and picks the one that best contains the query while rewarding surrounding context:

```js
const { seq, sentence } = require('nlp-lcs');

const query = sentence.tokenize('neural network training');
const paragraphs = article.split('\n\n');

const { value: bestParagraph } = seq.bestContext(
  query,
  paragraphs.map(p => sentence.tokenize(p)),
  sentence.lcs
);
// → The paragraph that best contains "neural network training" with the
//   most useful surrounding context.
```

The context scoring means a paragraph that mentions the phrase inside a rich explanation outranks one that merely contains it in a heading.

### 5. Typo-tolerant word replacement for BERT prompts

BERT-style models rely on exact vocabulary tokens. When user input contains typos, you can fuzzy-match each word against the model's vocabulary and replace with the closest known token:

```js
const { word } = require('nlp-lcs');

const bertVocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'];

const userInput = 'the quikc borwn fox jmups over the lzy dog';

const corrected = userInput.split(/\s+/).map(tok => {
  const { value, match } = word.bestMatch(tok, bertVocab);
  return match ? value : tok; // keep original if nothing is close enough
}).join(' ');
// → 'the quick brown fox jumps over the lazy dog'
```

The 80 % character-level threshold is lenient enough to catch common typos ("quikc" → "quick") but strict enough to reject unrelated words, so the prompt stays semantically faithful to user intent.

---

## How the threshold works

All match/has functions use an **80 % rule** derived from the Pareto distribution:

- **`match`** — score ≥ `floor(0.8 × max(len1, len2))` → "are these substantially similar?"
- **`has`** — score ≥ `floor(0.8 × min(len1, len2))` → "does one substantially appear within the other?"

The floor + max/min scaling gives short sequences proportionally more slack (e.g. "a" matches "an") while requiring longer sequences to align more closely.

## Early termination

`seq.lcs` accepts an optional `minScore` parameter (number or `(seq1, seq2) → number` function). When supplied, the DP loop checks after each row whether the required score is still mathematically reachable. If not, it returns early with a lower-bound score. Early-return results are intentionally **not cached** so future calls without a threshold still compute the exact answer.

## Memoization

Each comparator function gets its own LRU cache (default capacity 4096). Natural language has a small effective vocabulary relative to input volume, so cache hit rates are high — especially in the recursive `sentence.lcs` case where every outer DP cell triggers inner character-level comparisons on the same word pairs.

## License

MIT
