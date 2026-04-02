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

/**
 * Module-level memoization cache shared across all glcs calls.
 * Keys are the sorted, stringified pair of sequences, making comparison
 * order-independent. Hit rates are high for NLP workloads because natural
 * language has a small effective vocabulary relative to input volume —
 * the same word pairs recur constantly, especially in the recursive LCWS case
 * where every outer DP cell triggers an inner character-level LCS.
 */
const lcsMemo = new Map();

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
 * Uint8Array is used for the DP table rows — adequate for typical NLP sequence
 * lengths and cache-friendlier than a nested plain array.
 *
 * @param {Iterable} seq1
 * @param {Iterable} seq2
 * @param {function} compare - Element comparator. Defaults to strict equality.
 *   Swap in an LCS-based matcher to get recursive/hierarchical LCS.
 * @returns {number} Length of the longest common subsequence
 */
const glcs = function generalLongestCommonSubsequence(seq1, seq2, compare=(x,y)=>(x===y)) {
  "use strict";
  const key = String([seq1,seq2].sort());
  if(lcsMemo.has(key)){
    return lcsMemo.get(key);
  }
  let array1 = [...seq1??[]];
  let array2 = [...seq2??[]];
  // Always put the longer sequence on the outer dimension for consistency
  if (array2.length > array1.length) {
    [array1, array2] = [array2, array1];
  }
  const [arr1, arr2] = [array1, array2];
  const arr1_length = arr1.length;
  const arr2_length = arr2.length;
  const dp = Array(arr1_length + 1).fill(0).map(() => new Uint8Array(arr2_length + 1));
  const dp_length = dp.length;         // arr1_length + 1, outer dimension
  const dp_inner_length = arr2_length + 1; // inner dimension — distinct from dp_length
  for (let i = 1; i !== dp_length; ++i) {
    for (let x = 1; x !== dp_inner_length; ++x) { // was incorrectly dp_length — over-iterated inner dimension when lengths differ
      if (compare(arr1[i - 1], arr2[x - 1])) {
        dp[i][x] = dp[i - 1][x - 1] + 1;
      } else {
        dp[i][x] = Math.max(dp[i][x - 1], dp[i - 1][x]);
      }
    }
  }
  const score = dp[arr1_length][arr2_length];
  lcsMemo.set(key, score);
  return score;
};

/**
 * Normalize a value to a Unicode-decomposed, lowercased string.
 * Applied before character-level LCS so that accented variants, ligatures,
 * and case differences don't inflate the edit distance.
 *
 * @param {*} x
 * @returns {string}
 */
const norm = x => String(x).normalize('NFD').toLowerCase();

/**
 * Normalized LCS — character-level LCS on unicode-normalized, lowercased strings.
 * Base comparator for word-level matching; feeds into wordMatch and sentenceLcs.
 *
 * @param {string} str1
 * @param {string} str2
 * @returns {number}
 */
const nlcs = function normalizedLongestCommonSubsequence(str1, str2){
  return glcs(norm(str1), norm(str2));
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
const lcsMatch = (seq1, seq2, lcs=glcs) => {
  return lcs(seq1, seq2) >= Math.floor(0.8 * Math.max(seq1.length, seq2.length));
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
const lcsHas = (seq1, seq2, lcs=glcs) => {
  return lcs(seq1, seq2) >= Math.floor(0.8 * Math.min(seq1.length, seq2.length));
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
const weightedLcs = (seq1, seq2, lcs=glcs) => {
  return lcs(seq1, seq2) * Math.min(seq1.length, seq2.length) / Math.max(seq1.length, seq2.length, 1);
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
const contextLcs = (seq1, seq2, lcs=glcs) => {
  return lcs(seq1, seq2) + Math.max(seq1.length, seq2.length) / (Math.min(seq1.length, seq2.length) || 1);
};
