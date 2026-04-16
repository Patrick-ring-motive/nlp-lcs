/**
 * nlp-lcs — LCS-based sequence matching utilities for NLP pipelines
 *
 * Standard edit-distance metrics (Levenshtein) measure transformation cost,
 * which suits typo correction but penalizes legitimate paraphrases. LCS measures
 * shared skeleton — the longest subsequence preserved across both inputs — which
 * is more semantically load-bearing for natural language comparison.
 *
 * The variants here compose into a recursive hierarchy:
 *   character-level → word-level (LCWS) → sentence-level → ...
 * with memoization making the recursive cases tractable via high cache hit rates
 * on repeated vocabulary across a corpus.
 */
const bigramIntersection = (seq1,seq2)=>{
  let score = 0;
  if(seq1.length < seq2.length){
    [seq2,seq1] = [seq1,seq2];
  }
  const set1 = new Map();
  const set2 = new Map();
  const seq1_length = seq1.length + 1;
  const seq2_length = seq2.length + 1;
  for(let i = 0; i !== seq1_length; ++i){
    const key = String(seq1[i-1]) + String(seq1[i]);
    let count  = set1.get(key) ?? 0;
    set1.set(key,count+1);    
  }
  for(let i = 0; i !== seq2_length; ++i){
    const key = String(seq2[i-1]) + String(seq2[i]);
    let count  = set2.get(key) ?? 0;
    set2.set(key,count+1);    
  }
  for(const [key,value] of set1){
    score += Math.min(value,set2.get(key)??0);
  }
  return score;
};
/**
 * Get the length of any value.
 * Tries `.length`, then `.size()`, then `parseInt` as a numeric fallback.
 * Returns 0 when the value is null, undefined, or has no recognizable length.
 *
 * @param {*} x - Value to measure.
 * @returns {number}
 */
const len = x => x?.length || x?.size?.() || parseInt(x) || 0;

/**
 * Default element comparator for plain LCS.
 * Kept as a stable function reference so memo buckets keyed by comparator
 * can be reused across calls.
 */
const defaultCompare = (x, y) => (x === y);

/**
 * Simple LRU cache backed by insertion-ordered Map.
 * `get` refreshes recency by reinserting the key.
 */
const createLruCache = (limit = 4096) => {
  const map = new Map();
  return {
    get(key) {
      if (!map.has(key)) return undefined;
      const value = map.get(key);
      map.delete(key);
      map.set(key, value);
      return value;
    },
    set(key, value) {
      if (map.has(key)) map.delete(key);
      map.set(key, value);
      if (map.size > limit) {
        const oldestKey = map.keys().next().value;
        map.delete(oldestKey);
      }
      return value;
    }
  };
};

/**
 * Check whether a value is a string primitive or String object.
 *
 * @param {*} x - Value to test.
 * @returns {boolean}
 */
const isString = x => typeof x === 'string' || x instanceof String || x?.constructor?.name === 'String';

/**
 * Convert any value to a string representation.
 * Strings are returned as-is; other values are JSON-stringified.
 * Falls back to `String()` when JSON serialization throws (e.g. circular refs).
 *
 * @param {*} x - Value to stringify.
 * @returns {string}
 */
const stringify = x => {
  try {
    if(isString(x)) return String(x);
    return String(JSON.stringify(x));
  } catch {
    return String(x);
  } 
};

/**
 * One LRU cache per comparator function.
 * WeakMap keeps comparator-keyed buckets collectible when no longer referenced.
 */
const lcsMemoByCompare = new WeakMap();

/**
 * Retrieve (or create) the LRU cache associated with a comparator function.
 *
 * @param {function} compare - The comparator whose cache to retrieve.
 * @returns {{get: function(string): (number|undefined), set: function(string, number): number}}
 */
const getLcsMemo = (compare) => {
  let memo = lcsMemoByCompare.get(compare);
  if (!memo) {
    memo = createLruCache();
    lcsMemoByCompare.set(compare, memo);
  }
  return memo;
};

/**
 * Build an order-independent cache key from two sequences.
 * Both sequences are stringified and sorted lexicographically so that
 * `makeMemoKey(a, b) === makeMemoKey(b, a)`.
 *
 * @param {*} seq1 - First sequence.
 * @param {*} seq2 - Second sequence.
 * @returns {string}
 */
const makeMemoKey = (seq1, seq2) => {
  const str1 = stringify(seq1);
  const str2 = stringify(seq2);
  return (str1 <= str2) ? `${str1}|${str2}` : `${str2}|${str1}`;
};

/**
 * Select the smallest typed array whose element range can hold `maxValue`.
 * Falls back to a regular Array for values beyond Uint32 range.
 *
 * @param {number} maxValue - The largest value that will be stored.
 * @returns {typeof Uint8Array | typeof Uint16Array | typeof Uint32Array | typeof Array}
 */
const selectArrayType = (maxValue) => {
  if (maxValue <= 0xFF)       return Uint8Array;
  if (maxValue <= 0xFFFF)     return Uint16Array;
  if (maxValue <= 0xFFFFFFFF) return Uint32Array;
  return Array;
};

/**
 * General LCS — the foundational primitive.
 *
 * Standard DP formulation extended with a pluggable comparator, which is the
 * key abstraction that enables the recursive variants below. When `compare` is
 * the default strict equality, this is plain LCS. When `compare` is itself an
 * LCS-based matcher, you get hierarchical fuzzy alignment.
 *
 * Sequences are spread into arrays so the function accepts strings, arrays of
 * words, or any iterable. The longer sequence is always assigned to the outer
 * loop dimension to keep the DP table orientation consistent regardless of
 * argument order.
 *
 * The DP table element type is chosen by `selectArrayType` based on the shorter
 * sequence's length, keeping small inputs cache-friendly while still supporting
 * large inputs.
 *
 * When `minScore` is supplied (number or function), the DP loop checks after
 * each outer-row whether the remaining elements can still bridge the gap to
 * the required score. If not, the function returns early with the current
 * (partial, lower-bound) score. Early-return results are **not** cached so
 * future calls without a threshold still compute the exact answer.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} compare - Element comparator. Defaults to strict equality.
 *   Swap in an LCS-based matcher to get recursive/hierarchical LCS.
 * @param {number|function} [minScore=0] - Minimum useful score. If a number,
 *   used directly as the threshold. If a function, called as
 *   `minScore(seq1, seq2)` to compute the threshold. When the DP determines
 *   the threshold is unreachable it returns early with a lower-bound score.
 * @returns {number} Length of the longest common subsequence (exact when
 *   minScore is 0 or omitted; lower-bound on early termination).
 */
const glcs = function generalLongestCommonSubsequence(seq1, seq2, compare = defaultCompare, minScore = 0) {
  if(seq1 == null || seq2 == null) return 0;
  if (seq1 === seq2) {
    return seq1.length;
  }
  const cmp = (typeof compare === 'function') ? compare : defaultCompare;
  const lcsMemo = getLcsMemo(cmp);
  const key = makeMemoKey(seq1, seq2);
  const cachedScore = lcsMemo.get(key);
  if (cachedScore !== undefined) {
    return cachedScore;
  }

  // Resolve threshold — accept a number or a function(seq1, seq2) → number
  const threshold = (typeof minScore === 'function') ? minScore(seq1, seq2) : (minScore || 0);

  let array1 = len(seq1) ? seq1 : [...seq1 ?? []];
  let array2 = len(seq2) ? seq2 : [...seq2 ?? []];
  // Always put the longer sequence on the outer dimension for consistency
  if (array2.length > array1.length) {
    [array1, array2] = [array2, array1];
  }
  const [arr1, arr2] = [array1, array2];
  const arr1_length = arr1.length;
  const arr2_length = arr2.length;

  // Pre-DP rejection: the LCS can never exceed the shorter sequence's length
  if (threshold > 0 && arr2_length < threshold) {
    return new Set(arr1).intersection(new Set(arr2)); // upper-bound < threshold → can't match; return best-case
  }

  const DPArray = selectArrayType(arr2_length);
  const dp = Array(arr1_length + 1).fill(0).map(() => new DPArray(arr2_length + 1));
  const dp_length = dp.length; // arr1_length + 1, outer dimension
  const dp_inner_length = arr2_length + 1; // inner dimension — distinct from dp_length
  for (let i = 1; i !== dp_length; ++i) {
    for (let x = 1; x !== dp_inner_length; ++x) {
      if (arr1[i - 1] === arr2[x - 1]||arr1[i - 1] == arr2[x - 1]||cmp(arr1[i - 1], arr2[x - 1])) {
        dp[i][x] = dp[i - 1][x - 1] + 1;
      } else {
        dp[i][x] = Math.max(dp[i][x - 1], dp[i - 1][x]);
      }
    }
    // Early termination: best possible score from here can't reach the threshold
    if (threshold > 0) {
      const bestPossible = dp[i][arr2_length] + (arr1_length - i);
      if (bestPossible < threshold) {
        return dp[i][arr2_length]; // lower-bound; intentionally NOT cached
      }
    }
  }
  const score = dp[arr1_length][arr2_length];
  lcsMemo.set(key, score);
  return score;
};

/**
 * Extract the actual longest common subsequence (with backtracking).
 *
 * Builds the same DP table as `glcs` but then walks it backwards to recover
 * the matched elements. Returns both the score and the subsequence array.
 *
 * This is intentionally a separate function from `glcs` so the hot scoring
 * path is not burdened with backtracking overhead.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} [compare=defaultCompare] - Element comparator.
 * @returns {{score: number, subsequence: Array}} The LCS length and the
 *   matched elements drawn from `seq1` (or `seq2` when swapped to outer).
 */
const lcsSubsequence = (seq1, seq2, compare = defaultCompare) => {
  if (seq1 == null || seq2 == null) return { score: 0, subsequence: [] };
  if (seq1 === seq2) {
    return { score: seq1.length, subsequence: [...seq1] };
  }
  const cmp = (typeof compare === 'function') ? compare : defaultCompare;

  let array1 = len(seq1) ? seq1 : [...seq1 ?? []];
  let array2 = len(seq2) ? seq2 : [...seq2 ?? []];
  if (array2.length > array1.length) {
    [array1, array2] = [array2, array1];
  }
  const [arr1, arr2] = [array1, array2];
  const arr1_length = arr1.length;
  const arr2_length = arr2.length;

  const DPArray = selectArrayType(arr2_length);
  const dp = Array(arr1_length + 1).fill(0).map(() => new DPArray(arr2_length + 1));
  const dp_length = dp.length;
  const dp_inner_length = arr2_length + 1;
  for (let i = 1; i !== dp_length; ++i) {
    for (let x = 1; x !== dp_inner_length; ++x) {
      if (cmp(arr1[i - 1], arr2[x - 1])) {
        dp[i][x] = dp[i - 1][x - 1] + 1;
      } else {
        dp[i][x] = Math.max(dp[i][x - 1], dp[i - 1][x]);
      }
    }
  }

  // Backtrack to extract the subsequence
  const score = dp[arr1_length][arr2_length];
  const subsequence = [];
  let i = arr1_length;
  let x = arr2_length;
  while (i > 0 && x > 0) {
    if (cmp(arr1[i - 1], arr2[x - 1])) {
      subsequence.push(arr1[i - 1]);
      --i;
      --x;
    } else if (dp[i - 1][x] >= dp[i][x - 1]) {
      --i;
    } else {
      --x;
    }
  }
  subsequence.reverse();

  return { score, subsequence };
};

/**
 * Normalize a value to a Unicode-decomposed, lowercased string.
 * Applied before character-level LCS so that accented variants, ligatures,
 * and case differences don't inflate the edit distance.
 *
 * @param {*} x
 * @returns {string}
 */
const norm = x => stringify(x).trim().normalize('NFD').toLowerCase().trim();

/**
 * Normalized LCS — character-level LCS on unicode-normalized, lowercased strings.
 * Base comparator for word-level matching; feeds into wordMatch and sentenceLcs.
 *
 * @param {string} str1
 * @param {string} str2
 * @returns {number}
 */
const nlcs = function normalizedLongestCommonSubsequence(str1, str2) {
  return glcs(norm(str1), norm(str2));
};

/**
 * Test whether a raw LCS score meets the 80 % similarity threshold
 * relative to the *longer* of two sequences.
 *
 * @param {Iterable} seq1 - First sequence.
 * @param {Iterable} seq2 - Second sequence.
 * @param {number} [score=0] - Pre-computed LCS score.
 * @returns {boolean} `true` when `score >= floor(0.8 * max(len1, len2))`.
 */
const scoreMatch = (seq1, seq2, score=0) => {
  return score >= Math.floor(0.8 * (Math.max(len(seq1), len(seq2))||0));
};

/**
 * Threshold match using the Pareto-derived 80% rule.
 *
 * Returns true if the LCS score meets or exceeds 80% of the *longer* sequence's
 * length. The floor + max-length scaling is asymmetric by design: short sequences
 * get proportionally more slack (e.g. "a" matches "an"), while longer sequences
 * must align more closely. This mirrors human fuzzy-match intuition without a
 * magic number — 80/20 is grounded in Pareto distributions, which natural
 * language tends to follow.
 *
 * Use this when both sequences are peers and you want mutual similarity.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} lcs - LCS function to use. Defaults to glcs.
 * @returns {boolean}
 */
const lcsMatch = (seq1, seq2, lcs = glcs) => {
  return scoreMatch(seq1,seq2,lcs(seq1, seq2));
};

/**
 * Find the sequence in `seqList` with the highest LCS score against `seq1`.
 *
 * @param {Iterable} seq1 - The query sequence.
 * @param {Iterable[]} seqList - Candidate sequences to compare against.
 * @param {function} [lcs=glcs] - LCS scoring function.
 * @param {function} [matcher=scoreMatch] - Threshold function `(seq1, best, score) → boolean`.
 * @returns {{value: *, match: boolean, score: number}} Best candidate, whether it
 *   meets the threshold, and the raw score.
 */
const bestLcsMatch=(seq1,seqList,lcs=glcs,matcher=scoreMatch)=>{
  let score = -1;
  let value;
  let match = false;
  for(const seq2 of seqList){
    const matchScore = lcs(seq1,seq2);
    if(matchScore > score){
      score = matchScore;
      value = seq2;
    }
  }
  if (score < 0) return { value, match, score: 0 };
  match = matcher(seq1,value,score);
  return {value,match,score};
};

/**
 * Test whether a raw LCS score meets the 80 % containment threshold
 * relative to the *shorter* of two sequences.
 *
 * @param {Iterable} seq1 - First sequence.
 * @param {Iterable} seq2 - Second sequence.
 * @param {number} [score=0] - Pre-computed LCS score.
 * @returns {boolean} `true` when `score >= floor(0.8 * min(len1, len2))`.
 */
const scoreHas = (seq1, seq2, score=0) => {
  return score >= Math.floor(0.8 * (Math.min(len(seq1), len(seq2))||0));
};

/**
 * Containment match — lenient variant of lcsMatch.
 *
 * Thresholds against the *shorter* sequence rather than the longer. Answers:
 * "does seq1 substantially appear within seq2?" rather than "are seq1 and seq2
 * substantially similar?" Useful when seq1 is a query or phrase and seq2 is a
 * larger chunk you're scanning for it.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} lcs - LCS function to use. Defaults to glcs.
 * @returns {boolean}
 */
const lcsHas = (seq1, seq2, lcs = glcs) => {
  return scoreHas(seq1,seq2,lcs(seq1, seq2));
};

/**
 * Find the sequence in `seqList` with the highest LCS score against `seq1`,
 * using the containment (min-length) threshold.
 *
 * @param {Iterable} seq1 - The query sequence.
 * @param {Iterable[]} seqList - Candidate sequences.
 * @param {function} [lcs=glcs] - LCS scoring function.
 * @returns {{value: *, match: boolean, score: number}}
 */
const bestLcsHas=(seq1,seqList,lcs=glcs)=>{
  return bestLcsMatch(seq1,seqList,lcs,scoreHas);
};

/**
 * Word-level fuzzy match using character-level LCS as the comparator.
 *
 * Two words are considered equal if their normalized character-level LCS meets
 * the 80% threshold. Feeds into sentenceLcs as the element comparator.
 *
 * @param {string} seq1 - A single word
 * @param {string} seq2 - A single word
 * @returns {boolean}
 */
const wordMatch = (seq1, seq2) => {
  return lcsMatch(seq1, seq2, nlcs);
};

/**
 * Find the closest word in `seqList` to `seq1` using character-level fuzzy LCS.
 *
 * @param {string} seq1 - The query word.
 * @param {string[]} seqList - Candidate words.
 * @returns {{value: string, match: boolean, score: number}}
 */
const bestWordMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,nlcs,wordMatch);
};

/**
 * LCWS — Longest Common Word Subsequence (recursive / hierarchical LCS).
 *
 * Runs LCS over sequences of words where individual word comparison is done by
 * character-level fuzzy match (wordMatch) rather than strict equality. This
 * means structurally similar sentences align even when surface forms differ —
 * "he was running quickly" and "she ran fast" share more structure here than
 * pure string LCS or bag-of-words embeddings would capture.
 *
 * The memoization cache makes this tractable: the inner wordMatch calls are
 * expensive in isolation but natural language vocabulary is small relative to
 * corpus size, so the same word pairs resolve from cache repeatedly.
 *
 * Can be extended further: pass sentenceLcs as the comparator to glcs for
 * paragraph-level recursive alignment, and so on up the hierarchy.
 *
 * @param {string[]} words1 - Tokenized sentence
 * @param {string[]} words2 - Tokenized sentence
 * @returns {number}
 */
const sentenceLcs = (words1, words2) => {
  return glcs(words1, words2, wordMatch);
};



/**
 * Sentence-level fuzzy match using LCWS.
 * Returns true if the two tokenized sentences meet the 80% threshold at the
 * word-structure level.
 *
 * @param {string[]} seq1 - Tokenized sentence
 * @param {string[]} seq2 - Tokenized sentence
 * @returns {boolean}
 */
const sentenceMatch = (seq1, seq2) => {
  return lcsMatch(seq1, seq2, sentenceLcs);
};

/**
 * Find the most similar tokenized sentence in `seqList` to `seq1`
 * using word-level fuzzy LCS.
 *
 * @param {string[]} seq1 - Tokenized query sentence.
 * @param {string[][]} seqList - Candidate tokenized sentences.
 * @returns {{value: string[], match: boolean, score: number}}
 */
const bestSentenceMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,sentenceLcs,sentenceMatch);
};

/**
 * Weighted LCS — similarity score penalized by length disparity.
 *
 * Multiplies the raw LCS score by (min_length / max_length), a ratio that
 * approaches 1 for same-length sequences and shrinks toward 0 as lengths
 * diverge. Effectively a length-normalized score resembling Jaccard similarity.
 *
 * Use this as a reranker when you want matched sequences to be close in length
 * — i.e., you prefer precise paraphrases over containment.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} lcs - LCS function to use. Defaults to glcs.
 * @returns {number}
 */
const weightedLcs = (seq1, seq2, lcs = glcs) => {
  return lcs(seq1, seq2) * (Math.min(len(seq1), len(seq2))||0) / (Math.max(len(seq1), len(seq2), 1)||1);
};

/**
 * Find the best length-penalized LCS match in `seqList`.
 *
 * @param {Iterable} seq1 - The query sequence.
 * @param {Iterable[]} seqList - Candidate sequences.
 * @returns {{value: *, match: boolean, score: number}}
 */
const bestWeightedMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,weightedLcs);
};

/**
 * Weighted LCS at the character level using normalized comparison.
 * Combines `weightedLcs` with `nlcs` for single-word scoring.
 *
 * @param {string} seq1 - First word.
 * @param {string} seq2 - Second word.
 * @returns {number}
 */
const weightedWordLcs = (seq1, seq2) => {
  return weightedLcs(seq1,seq2,nlcs);
};

/**
 * Find the best length-penalized word match in `seqList` using
 * normalized character-level LCS.
 *
 * @param {string} seq1 - The query word.
 * @param {string[]} seqList - Candidate words.
 * @returns {{value: string, match: boolean, score: number}}
 */
const bestWeightedWordMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,weightedWordLcs);
};

/**
 * Context LCS — match score that rewards longer containing sequences.
 *
 * Adds a length bonus (max / min) to the raw LCS score rather than multiplying
 * by a penalty. A short query matching inside a long paragraph scores higher
 * than the same query matching inside a short sentence, because the longer
 * result carries more surrounding context.
 *
 * The additive form means length has weight but doesn't dominate — a paragraph
 * with a weak LCS match won't beat a sentence with a strong one just by being
 * long. Designed for scanning a document (e.g. a Wikipedia article) for the
 * most contextually useful paragraph containing a phrase.
 *
 * The `|| 1` guards against zero-length sequences on the denominator.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} lcs - LCS function to use. Defaults to glcs.
 * @returns {number}
 */
const contextLcs = (seq1, seq2, lcs = glcs) => {
  return lcs(seq1, seq2) + (Math.max(len(seq1), len(seq2))||0) / (Math.min(len(seq1), len(seq2)) || 1);
};

/**
 * Find the best context-rewarding LCS match in `seqList`.
 *
 * @param {Iterable} seq1 - The query sequence.
 * @param {Iterable[]} seqList - Candidate sequences.
 * @returns {{value: *, match: boolean, score: number}}
 */
const bestContextMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,contextLcs);
};

/**
 * Context LCS at the character level using normalized comparison.
 * Combines `contextLcs` with `nlcs` for single-word scoring that
 * rewards longer containing words.
 *
 * @param {string} seq1 - First word.
 * @param {string} seq2 - Second word.
 * @returns {number}
 */
const contextWordLcs = (seq1, seq2) => {
  return contextLcs(seq1, seq2, nlcs);
};

/**
 * Find the best context-rewarding word match in `seqList` using
 * normalized character-level LCS.
 *
 * @param {string} seq1 - The query word.
 * @param {string[]} seqList - Candidate words.
 * @returns {{value: string, match: boolean, score: number}}
 */
const bestContextWordMatch=(seq1,seqList)=>{
  return bestLcsMatch(seq1,seqList,contextWordLcs);
};

/**
 * Sørensen–Dice similarity using LCS.
 *
 * Computes 2 × LCS(seq1, seq2) / (len1 + len2), producing a value in [0, 1].
 * Unlike the 80 % threshold functions, this gives a continuous similarity
 * score that treats both sequences symmetrically and penalizes length
 * differences more gently than `weightedLcs`.
 *
 * Returns 1 for identical sequences, 0 when they share nothing, and 0
 * when both are empty (avoids 0/0).
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} lcs - LCS function to use. Defaults to glcs.
 * @returns {number} Similarity in [0, 1].
 */
const diceLcs = (seq1, seq2, lcs = glcs) => {
  const total = len(seq1) + len(seq2);
  if (total === 0) return 0;
  return (2 * lcs(seq1, seq2)) / total;
};

/**
 * Sørensen–Dice similarity at the character level using normalized comparison.
 *
 * @param {string} seq1 - First word.
 * @param {string} seq2 - Second word.
 * @returns {number} Similarity in [0, 1].
 */
const diceWordLcs = (seq1, seq2) => {
  return diceLcs(seq1, seq2, nlcs);
};

/**
 * Sørensen–Dice similarity at the sentence level using word-fuzzy LCS.
 *
 * @param {string[]} words1 - Tokenized sentence.
 * @param {string[]} words2 - Tokenized sentence.
 * @returns {number} Similarity in [0, 1].
 */
const diceSentenceLcs = (words1, words2) => {
  return diceLcs(words1, words2, sentenceLcs);
};


/**
 * Split a string into whitespace-delimited tokens.
 *
 * @param {string} str - Input text.
 * @returns {string[]} Array of tokens.
 */
const tokenize = str => str.trim().split(/\s+/);

/**
 * Split text on sentence-ending punctuation (`.`, `!`, `?`).
 * Empty segments are filtered out.
 *
 * @param {string} str - Input text.
 * @returns {string[]} Array of trimmed sentence strings.
 */
const toSentences = str => str.split(/[.!?]+/).map(s => s.trim()).filter(Boolean);

/**
 * Split text on phrase-level punctuation (`.`, `!`, `?`, `,`, `;`).
 * Empty segments are filtered out.
 *
 * @param {string} str - Input text.
 * @returns {string[]} Array of trimmed phrase strings.
 */
const toPhrases = str => str.split(/[.!?,;]+/).map(s => s.trim()).filter(Boolean);

/**
 * Remove fuzzy-duplicate text segments using sentence-level LCS matching.
 * When two segments match, the longer one is kept by default.
 *
 * @param {string[]} segments - Text segments to deduplicate.
 * @param {object} [options]
 * @param {boolean} [options.preferLonger=true] - Keep the longer of two
 *   matching segments. When `false`, the first-seen segment is kept.
 * @returns {string[]} Deduplicated segments.
 */
const dedupeSegments = (segments, {
  preferLonger = true,
  tokenizer = tokenize,
} = {}) => {
  const kept = [];
  for (const s of segments) {
    const tokens = tokenizer(s);
    const matchIdx = kept.findIndex(k => sentenceMatch(tokenizer(k), tokens));
    if (matchIdx === -1) {
      kept.push(s);
    } else if (preferLonger && s.length > kept[matchIdx].length) {
      kept[matchIdx] = s;
    }
  }
  return kept;
};

/**
 * Split text into sentences and remove fuzzy duplicates.
 *
 * @param {string} text - Input text.
 * @param {object} [opts] - Options forwarded to `dedupeSegments`.
 * @returns {string[]} Deduplicated sentence strings.
 */
const dedupeSentences = (text, opts) => dedupeSegments(toSentences(text), opts);

/**
 * Split text into phrases and remove fuzzy duplicates.
 *
 * @param {string} text - Input text.
 * @param {object} [opts] - Options forwarded to `dedupeSegments`.
 * @returns {string[]} Deduplicated phrase strings.
 */
const dedupePhrases = (text, opts) => dedupeSegments(toPhrases(text), opts);

/**
 * Compute LCS on sorted copies of both sequences.
 * Sorting removes order dependence, making this useful for bag-like comparison.
 *
 * @param {Iterable} seq1 - First sequence (not mutated).
 * @param {Iterable} seq2 - Second sequence (not mutated).
 * @param {function} [lcs=glcs] - LCS scoring function.
 * @returns {number}
 */
const sortedLcs = (seq1, seq2, lcs = glcs) => {
  return lcs([...seq1].sort(), [...seq2].sort());
};

/**
 * Sorted word-level LCS — tokenizes both strings, sorts the token arrays,
 * then computes sentence-level fuzzy LCS. Useful for order-insensitive
 * comparison of two sentences.
 *
 * @param {string} str1 - First sentence string.
 * @param {string} str2 - Second sentence string.
 * @returns {number}
 */
const sortedWordLcs = (str1, str2) => {
  return sortedLcs(tokenize(str1), tokenize(str2), sentenceLcs);
};

/* ──────────────────────────────────────────────────────────────────────────
 * Namespaces — group functions by the sequence granularity they operate on.
 *   seq      → general / arbitrary-element sequences
 *   word     → character-level (single-word) comparisons
 *   sentence → tokenized-sentence (word-sequence) comparisons
 * ────────────────────────────────────────────────────────────────────────── */

const seq = {
  lcs: glcs,
  subsequence: lcsSubsequence,
  scoreMatch,
  match: lcsMatch,
  bestMatch: bestLcsMatch,
  scoreHas,
  has: lcsHas,
  bestHas: bestLcsHas,
  weighted: weightedLcs,
  bestWeighted: bestWeightedMatch,
  context: contextLcs,
  bestContext: bestContextMatch,
  sorted: sortedLcs,
  dice: diceLcs,
};

const word = {
  norm,
  lcs: nlcs,
  match: wordMatch,
  bestMatch: bestWordMatch,
  weighted: weightedWordLcs,
  bestWeighted: bestWeightedWordMatch,
  context: contextWordLcs,
  bestContext: bestContextWordMatch,
  sorted: sortedWordLcs,
  dice: diceWordLcs,
};

const sentence = {
  lcs: sentenceLcs,
  match: sentenceMatch,
  bestMatch: bestSentenceMatch,
  tokenize,
  toSentences,
  toPhrases,
  dedupeSegments,
  dedupeSentences,
  dedupePhrases,
  dice: diceSentenceLcs,
};

module.exports = {
  len,
  defaultCompare,
  createLruCache,
  isString,
  stringify,
  makeMemoKey,
  selectArrayType,
  seq,
  word,
  sentence,
};
