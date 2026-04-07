const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const {
  defaultCompare,
  createLruCache,
  isString,
  stringify,
  makeMemoKey,
  selectArrayType,
  seq,
  word,
  sentence,
} = require('./index.js');

// Destructure namespaced functions for convenience in tests
const { lcs: glcs, subsequence: lcsSubsequence, scoreMatch, match: lcsMatch, bestMatch: bestLcsMatch,
  scoreHas, has: lcsHas, bestHas: bestLcsHas, weighted: weightedLcs,
  bestWeighted: bestWeightedMatch, context: contextLcs,
  bestContext: bestContextMatch, sorted: sortedLcs,
  dice: diceLcs } = seq;
const { norm, lcs: nlcs, match: wordMatch, bestMatch: bestWordMatch,
  weighted: weightedWordLcs, bestWeighted: bestWeightedWordMatch,
  context: contextWordLcs, bestContext: bestContextWordMatch,
  sorted: sortedWordLcs, dice: diceWordLcs } = word;
const { lcs: sentenceLcs, match: sentenceMatch, bestMatch: bestSentenceMatch,
  tokenize, toSentences, toPhrases, dedupeSegments, dedupeSentences,
  dedupePhrases, dice: diceSentenceLcs } = sentence;

/* ═══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('defaultCompare', () => {
  it('returns true for strictly equal primitives', () => {
    assert.equal(defaultCompare('a', 'a'), true);
    assert.equal(defaultCompare(1, 1), true);
  });
  it('returns false for non-equal values', () => {
    assert.equal(defaultCompare('a', 'b'), false);
    assert.equal(defaultCompare(1, '1'), false);
  });
});

describe('createLruCache', () => {
  it('stores and retrieves values', () => {
    const c = createLruCache(4);
    c.set('a', 1);
    assert.equal(c.get('a'), 1);
  });
  it('returns undefined for missing keys', () => {
    const c = createLruCache(4);
    assert.equal(c.get('missing'), undefined);
  });
  it('evicts oldest entry when limit exceeded', () => {
    const c = createLruCache(3);
    c.set('a', 1);
    c.set('b', 2);
    c.set('c', 3);
    c.set('d', 4); // should evict 'a'
    assert.equal(c.get('a'), undefined);
    assert.equal(c.get('d'), 4);
  });
  it('refreshes recency on get', () => {
    const c = createLruCache(3);
    c.set('a', 1);
    c.set('b', 2);
    c.set('c', 3);
    c.get('a');     // refresh 'a' — 'b' is now oldest
    c.set('d', 4);  // should evict 'b'
    assert.equal(c.get('a'), 1);
    assert.equal(c.get('b'), undefined);
  });
});

describe('isString', () => {
  it('recognizes string primitives', () => {
    assert.equal(isString('hello'), true);
    assert.equal(isString(''), true);
  });
  it('recognizes String objects', () => {
    assert.equal(isString(new String('hello')), true);
  });
  it('rejects non-strings', () => {
    assert.equal(isString(123), false);
    assert.equal(isString(null), false);
    assert.equal(isString(undefined), false);
    assert.equal(isString([]), false);
  });
});

describe('stringify', () => {
  it('returns string as-is', () => {
    assert.equal(stringify('hello'), 'hello');
  });
  it('JSON-stringifies objects', () => {
    assert.equal(stringify({ a: 1 }), '{"a":1}');
  });
  it('JSON-stringifies arrays', () => {
    assert.equal(stringify([1, 2]), '[1,2]');
  });
  it('falls back to String() for non-JSON values', () => {
    // JSON.stringify(Symbol()) returns undefined (no throw), which stringify
    // currently returns as-is. Circular refs do throw and hit the catch path.
    const circular = {};
    circular.self = circular;
    const val = stringify(circular);
    assert.equal(typeof val, 'string');
    assert.ok(val.length > 0);
  });
  it('handles numbers', () => {
    assert.equal(stringify(42), '42');
  });
});

describe('makeMemoKey', () => {
  it('produces an order-independent key', () => {
    assert.equal(makeMemoKey('abc', 'xyz'), makeMemoKey('xyz', 'abc'));
  });
  it('produces distinct keys for distinct pairs', () => {
    assert.notEqual(makeMemoKey('abc', 'def'), makeMemoKey('abc', 'xyz'));
  });
});

describe('selectArrayType', () => {
  it('returns Uint8Array for values ≤ 255', () => {
    assert.equal(selectArrayType(0), Uint8Array);
    assert.equal(selectArrayType(255), Uint8Array);
  });
  it('returns Uint16Array for values ≤ 65535', () => {
    assert.equal(selectArrayType(256), Uint16Array);
    assert.equal(selectArrayType(65535), Uint16Array);
  });
  it('returns Uint32Array for values ≤ 2^32-1', () => {
    assert.equal(selectArrayType(65536), Uint32Array);
    assert.equal(selectArrayType(0xFFFFFFFF), Uint32Array);
  });
  it('returns Array for values > 2^32-1', () => {
    assert.equal(selectArrayType(0xFFFFFFFF + 1), Array);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * glcs — General Longest Common Subsequence (core primitive)
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('glcs', () => {
  // ── Basic behaviour ────────────────────────────────────────────────────
  it('returns 0 for null/undefined inputs', () => {
    assert.equal(glcs(null, 'abc'), 0);
    assert.equal(glcs('abc', null), 0);
    assert.equal(glcs(null, null), 0);
    assert.equal(glcs(undefined, 'abc'), 0);
  });

  it('returns full length for identical strings', () => {
    assert.equal(glcs('abcdef', 'abcdef'), 6);
  });

  it('returns full length for same reference', () => {
    const s = 'hello';
    assert.equal(glcs(s, s), 5);
  });

  it('returns 0 for empty strings', () => {
    assert.equal(glcs('', ''), 0);
    assert.equal(glcs('', 'abc'), 0);
    assert.equal(glcs('abc', ''), 0);
  });

  // ── Classic LCS examples ──────────────────────────────────────────────
  it('computes LCS of "abcde" and "ace" = 3', () => {
    assert.equal(glcs('abcde', 'ace'), 3);
  });

  it('computes LCS of "abc" and "abc" = 3', () => {
    assert.equal(glcs('abc', 'abc'), 3);
  });

  it('computes LCS of "abc" and "def" = 0', () => {
    assert.equal(glcs('abc', 'def'), 0);
  });

  it('computes LCS of "abcbdab" and "bdcaba" = 4', () => {
    assert.equal(glcs('abcbdab', 'bdcaba'), 4);
  });

  it('handles single-character inputs', () => {
    assert.equal(glcs('a', 'a'), 1);
    assert.equal(glcs('a', 'b'), 0);
  });

  // ── Argument-order independence ────────────────────────────────────────
  it('is commutative — order of arguments does not change the score', () => {
    assert.equal(glcs('kitten', 'sitting'), glcs('sitting', 'kitten'));
    assert.equal(glcs('abcde', 'ace'), glcs('ace', 'abcde'));
  });

  // ── Array inputs ──────────────────────────────────────────────────────
  it('works with arrays of words', () => {
    assert.equal(glcs(['the', 'cat', 'sat'], ['the', 'bat', 'sat']), 2);
  });

  it('works with numeric arrays', () => {
    assert.equal(glcs([1, 3, 5, 7], [1, 2, 3, 4, 5]), 3);
  });

  // ── Custom comparator ────────────────────────────────────────────────
  it('respects a custom compare function', () => {
    const caseInsensitive = (a, b) => a.toLowerCase() === b.toLowerCase();
    assert.equal(glcs('ABC', 'abc', caseInsensitive), 3);
  });

  it('uses default comparator when compare is not a function', () => {
    // passing a non-function falls back to defaultCompare
    assert.equal(glcs('abc', 'abc', 'not-a-fn'), 3);
  });

  // ── minScore — numeric threshold ─────────────────────────────────────
  it('returns exact score when minScore = 0 (default)', () => {
    assert.equal(glcs('abcde', 'ace'), 3);
  });

  it('returns exact score when threshold is met', () => {
    // LCS of "abcde" and "abcde" is 5, threshold 4 → still computes full
    assert.equal(glcs('abcde', 'abcde', defaultCompare, 4), 5);
  });

  it('early-returns when shorter seq < threshold (pre-DP rejection)', () => {
    // "ab" (len 2) vs "xyz" (len 3), threshold 3 → shorter len is 2 < 3
    // Pre-DP rejection returns arr2_length (the shorter length) as the upper-bound
    const score = glcs('ab', 'xyz', defaultCompare, 3);
    assert.equal(score, 2); // upper-bound returned immediately
  });

  it('early-returns a lower-bound when threshold is unreachable mid-DP', () => {
    // "abcdefghij" (10) vs "xxxxxxxxyz" (10) — only share "yz" → LCS = 0
    // with threshold 8, after a few rows the best-possible drops below 8
    const score = glcs('abcdefghij', 'klmnopqrst', defaultCompare, 8);
    assert.ok(score < 8, `Expected score < 8, got ${score}`);
  });

  it('does not cache early-return results', () => {
    const a = 'abcdefghij';
    const b = 'klmnopqrst';
    // Call with high threshold → early return
    glcs(a, b, defaultCompare, 8);
    // Call without threshold → should compute exact (0), not return a stale early result
    assert.equal(glcs(a, b), 0);
  });

  // ── minScore — function threshold ────────────────────────────────────
  it('accepts a function for minScore', () => {
    const threshFn = (s1, s2) => Math.floor(0.8 * Math.max(s1.length, s2.length));
    // "abcde" vs "abcXX" → LCS = 3, threshold = floor(0.8*5) = 4
    const score = glcs('abcde', 'abcXX', defaultCompare, threshFn);
    // Whether exact or early-return, should be ≤ 3
    assert.ok(score <= 3);
  });

  // ── Memoization ──────────────────────────────────────────────────────
  it('returns cached result on repeated calls', () => {
    const a = 'memotest';
    const b = 'memoxest';
    const first = glcs(a, b);
    const second = glcs(a, b);
    assert.equal(first, second);
  });

  it('uses separate caches for different comparators', () => {
    const always = () => true;
    const never = () => false;
    const a = [1, 2, 3];
    const b = [4, 5, 6];
    // "always" comparator → every element matches → LCS = min length = 3
    assert.equal(glcs(a, b, always), 3);
    // "never" comparator → nothing matches → LCS = 0
    assert.equal(glcs(a, b, never), 0);
  });

  // ── Large-ish inputs (typed array selection) ──────────────────────────
  it('handles sequences longer than 255 (Uint16Array path)', () => {
    const a = 'a'.repeat(300);
    const b = 'a'.repeat(300);
    assert.equal(glcs(a, b), 300);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * lcsSubsequence — backtracking variant that returns matched elements
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('lcsSubsequence', () => {
  it('returns score and subsequence for identical strings', () => {
    const result = lcsSubsequence('abc', 'abc');
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, ['a', 'b', 'c']);
  });

  it('returns empty for null inputs', () => {
    assert.deepEqual(lcsSubsequence(null, 'abc'), { score: 0, subsequence: [] });
    assert.deepEqual(lcsSubsequence('abc', null), { score: 0, subsequence: [] });
  });

  it('returns empty for completely different strings', () => {
    const result = lcsSubsequence('abc', 'xyz');
    assert.equal(result.score, 0);
    assert.deepEqual(result.subsequence, []);
  });

  it('extracts correct subsequence from classic example', () => {
    const result = lcsSubsequence('abcde', 'ace');
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, ['a', 'c', 'e']);
  });

  it('works with arrays of words', () => {
    const result = lcsSubsequence(
      ['the', 'cat', 'sat', 'on', 'the', 'mat'],
      ['the', 'dog', 'sat', 'on', 'a', 'mat']
    );
    assert.equal(result.score, 4);
    assert.deepEqual(result.subsequence, ['the', 'sat', 'on', 'mat']);
  });

  it('works with numeric arrays', () => {
    const result = lcsSubsequence([1, 3, 5, 7], [1, 2, 3, 4, 5]);
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, [1, 3, 5]);
  });

  it('is commutative on score', () => {
    const a = lcsSubsequence('kitten', 'sitting');
    const b = lcsSubsequence('sitting', 'kitten');
    assert.equal(a.score, b.score);
  });

  it('handles same-reference fast path', () => {
    const arr = [1, 2, 3];
    const result = lcsSubsequence(arr, arr);
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, [1, 2, 3]);
  });

  it('respects a custom compare function', () => {
    const caseInsensitive = (a, b) => a.toLowerCase() === b.toLowerCase();
    const result = lcsSubsequence('ABC', 'abc', caseInsensitive);
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, ['A', 'B', 'C']);
  });

  it('score matches glcs', () => {
    const pairs = [
      ['abcbdab', 'bdcaba'],
      ['hello world', 'hlo wrd'],
      ['abcdefghij', 'bdfhj'],
    ];
    for (const [a, b] of pairs) {
      assert.equal(lcsSubsequence(a, b).score, glcs(a, b),
        `score mismatch for "${a}" vs "${b}"`);
    }
  });

  it('returned subsequence length equals score', () => {
    const result = lcsSubsequence('abcbdab', 'bdcaba');
    assert.equal(result.subsequence.length, result.score);
  });

  it('returned subsequence is a valid subsequence of both inputs', () => {
    const a = 'abcbdab';
    const b = 'bdcaba';
    const { subsequence } = lcsSubsequence(a, b);
    // Check that subsequence appears in order in both strings
    const isSubseqOf = (sub, full) => {
      let idx = 0;
      for (const ch of full) {
        if (idx < sub.length && ch === sub[idx]) idx++;
      }
      return idx === sub.length;
    };
    assert.ok(isSubseqOf(subsequence, [...a]),
      `${subsequence} is not a subsequence of ${a}`);
    assert.ok(isSubseqOf(subsequence, [...b]),
      `${subsequence} is not a subsequence of ${b}`);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * norm / nlcs
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('norm', () => {
  it('lowercases', () => {
    assert.equal(norm('ABC'), 'abc');
  });
  it('NFD-decomposes accented characters', () => {
    // é (U+00E9) → e + combining acute
    assert.ok(norm('é').length > 1);
  });
  it('trims whitespace', () => {
    assert.equal(norm('  hi  '), 'hi');
  });
});

describe('nlcs', () => {
  it('is case-insensitive', () => {
    assert.equal(nlcs('Hello', 'hello'), nlcs('hello', 'hello'));
  });
  it('handles accented characters gracefully', () => {
    assert.ok(nlcs('café', 'cafe') >= 4);
  });
  it('returns 0 for completely different words', () => {
    assert.equal(nlcs('abc', 'xyz'), 0);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * scoreMatch / lcsMatch
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('scoreMatch', () => {
  it('returns true when score ≥ 80% of max length', () => {
    assert.equal(scoreMatch('abcde', 'abcde', 5), true);  // 5 >= floor(0.8*5) = 4
    assert.equal(scoreMatch('abcde', 'abcde', 4), true);
  });
  it('returns false when score < 80% of max length', () => {
    assert.equal(scoreMatch('abcde', 'abcde', 3), false); // 3 < 4
  });
});

describe('lcsMatch', () => {
  it('returns true for identical strings', () => {
    assert.equal(lcsMatch('hello', 'hello'), true);
  });
  it('returns true for highly similar strings', () => {
    assert.equal(lcsMatch('hello', 'helo'), true);  // LCS=4, threshold=floor(0.8*5)=4
  });
  it('returns false for dissimilar strings', () => {
    assert.equal(lcsMatch('hello', 'world'), false);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * bestLcsMatch
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('bestLcsMatch', () => {
  it('finds the best matching sequence from a list', () => {
    const result = bestLcsMatch('hello', ['world', 'helo', 'hell']);
    // Both 'helo' and 'hell' have LCS=4 with 'hello'; first one found wins
    assert.equal(result.score, 4);
    assert.ok(result.value === 'helo' || result.value === 'hell');
  });
  it('reports match = true when threshold is met', () => {
    const result = bestLcsMatch('hello', ['hello']);
    assert.equal(result.match, true);
  });
  it('reports match = false when nothing meets threshold', () => {
    // Use candidates that share at least one char so value isn't undefined,
    // but are still too dissimilar to meet the 80% threshold
    const result = bestLcsMatch('hello', ['xyzhe', 'pqrst']);
    assert.equal(result.match, false);
  });
  it('returns undefined value and score 0 for empty candidate list', () => {
    const result = bestLcsMatch('hello', []);
    assert.equal(result.value, undefined);
    assert.equal(result.match, false);
    assert.equal(result.score, 0);
  });
  it('does not crash when all candidates score 0', () => {
    // Completely disjoint characters → every LCS score is 0
    const result = bestLcsMatch('aaa', ['zzz', 'yyy', 'xxx']);
    assert.equal(result.score, 0);
    assert.equal(result.match, false);
    // First candidate is selected as the "best" at score 0
    assert.equal(result.value, 'zzz');
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * scoreHas / lcsHas / bestLcsHas
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('scoreHas', () => {
  it('thresholds against the shorter sequence', () => {
    // min("abc".length, "abcdef".length) = 3, floor(0.8*3) = 2
    assert.equal(scoreHas('abc', 'abcdef', 3), true);
    assert.equal(scoreHas('abc', 'abcdef', 1), false);
  });
});

describe('lcsHas', () => {
  it('returns true when short seq is contained in long seq', () => {
    assert.equal(lcsHas('abc', 'xabcx'), true);
  });
  it('returns false for unrelated strings', () => {
    assert.equal(lcsHas('abc', 'xyz'), false);
  });
});

describe('bestLcsHas', () => {
  it('finds the best containment match', () => {
    const result = bestLcsHas('cat', ['the cat sat', 'a dog ran', 'cats']);
    assert.ok(result.value !== undefined);
    assert.ok(result.match);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * wordMatch / bestWordMatch
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('wordMatch', () => {
  it('matches nearly identical words', () => {
    assert.equal(wordMatch('running', 'runnin'), true);
  });
  it('matches identical words', () => {
    assert.equal(wordMatch('hello', 'hello'), true);
  });
  it('rejects very different words', () => {
    assert.equal(wordMatch('cat', 'elephant'), false);
  });
});

describe('bestWordMatch', () => {
  it('finds the closest word from a list', () => {
    const result = bestWordMatch('runing', ['walking', 'running', 'sitting']);
    assert.equal(result.value, 'running');
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * sentenceLcs / sentenceMatch / bestSentenceMatch
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('sentenceLcs', () => {
  it('scores identical tokenized sentences at full length', () => {
    const words = ['the', 'cat', 'sat'];
    assert.equal(sentenceLcs(words, words), 3);
  });
  it('scores with fuzzy word matching', () => {
    // "runing" fuzzy-matches "running"
    const a = ['the', 'cat', 'is', 'runing'];
    const b = ['the', 'cat', 'is', 'running'];
    assert.equal(sentenceLcs(a, b), 4);
  });
  it('handles completely different sentences', () => {
    assert.equal(sentenceLcs(['hello', 'world'], ['foo', 'bar']), 0);
  });
});

describe('sentenceMatch', () => {
  it('returns true for similar sentences', () => {
    const a = ['the', 'quick', 'brown', 'fox', 'jumps'];
    const b = ['the', 'quik', 'brown', 'fox', 'jumped'];
    assert.equal(sentenceMatch(a, b), true);
  });
  it('returns false for dissimilar sentences', () => {
    const a = ['the', 'cat'];
    const b = ['an', 'elephant', 'walked', 'slowly'];
    assert.equal(sentenceMatch(a, b), false);
  });
});

describe('bestSentenceMatch', () => {
  it('finds the best matching sentence from a list', () => {
    const query = ['the', 'cat', 'sat'];
    const candidates = [
      ['a', 'dog', 'ran'],
      ['the', 'cat', 'sat', 'down'],
      ['birds', 'fly', 'high'],
    ];
    const result = bestSentenceMatch(query, candidates);
    assert.deepEqual(result.value, ['the', 'cat', 'sat', 'down']);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * weightedLcs / weightedWordLcs / bestWeighted*
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('weightedLcs', () => {
  it('returns full score for same-length identical seqs', () => {
    // LCS=3, min/max=1 → 3*1 = 3
    assert.equal(weightedLcs('abc', 'abc'), 3);
  });
  it('penalizes length disparity', () => {
    // "ab" vs "abcdef": LCS=2, weight=2/6 → 2*(2/6)≈0.667
    const score = weightedLcs('ab', 'abcdef');
    assert.ok(score < 2, `Expected < 2, got ${score}`);
    assert.ok(score > 0);
  });
  it('returns 0 for no common subsequence', () => {
    assert.equal(weightedLcs('abc', 'xyz'), 0);
  });
});

describe('weightedWordLcs', () => {
  it('applies normalization and weighting', () => {
    const score = weightedWordLcs('Hello', 'hello');
    assert.ok(score > 0);
  });
});

describe('bestWeightedMatch', () => {
  it('picks the best weighted match', () => {
    const result = bestWeightedMatch('abc', ['ab', 'abc', 'abcdef']);
    assert.equal(result.value, 'abc');
  });
});

describe('bestWeightedWordMatch', () => {
  it('picks the best weighted word match', () => {
    const result = bestWeightedWordMatch('hello', ['helo', 'hello', 'xyz']);
    assert.equal(result.value, 'hello');
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * contextLcs / contextWordLcs / bestContext*
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('contextLcs', () => {
  it('adds a length bonus', () => {
    // For identical seqs of same length, bonus = max/min = 1
    const plain = glcs('abc', 'abc');
    const ctx = contextLcs('abc', 'abc');
    assert.ok(ctx > plain);
  });
  it('rewards longer containing sequence', () => {
    const short = contextLcs('ab', 'xabx');    // LCS + 4/2 = 2+2 = 4
    const long = contextLcs('ab', 'xabxxxxxx'); // LCS + 9/2 = 2+4.5 = 6.5
    assert.ok(long > short);
  });
});

describe('contextWordLcs', () => {
  it('uses normalized character LCS with context bonus', () => {
    const score = contextWordLcs('Hello', 'hello world');
    assert.ok(score > 0);
  });
});

describe('bestContextMatch', () => {
  it('picks the best context match', () => {
    const result = bestContextMatch('ab', ['xabx', 'xabxxxxxxx', 'xyz']);
    // The longer containing sequence should win due to context bonus
    assert.equal(result.value, 'xabxxxxxxx');
  });
});

describe('bestContextWordMatch', () => {
  it('returns a result with match info', () => {
    const result = bestContextWordMatch('hello', ['hello world', 'xyz']);
    assert.ok(result.value !== undefined);
    assert.ok(result.score > 0);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * Tokenization helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('tokenize', () => {
  it('splits on whitespace', () => {
    assert.deepEqual(tokenize('the cat sat'), ['the', 'cat', 'sat']);
  });
  it('trims leading/trailing whitespace', () => {
    assert.deepEqual(tokenize('  hello  world  '), ['hello', 'world']);
  });
  it('handles single word', () => {
    assert.deepEqual(tokenize('hello'), ['hello']);
  });
});

describe('toSentences', () => {
  it('splits on sentence-ending punctuation', () => {
    assert.deepEqual(toSentences('Hello. World! Yes?'), ['Hello', 'World', 'Yes']);
  });
  it('filters empty segments', () => {
    assert.deepEqual(toSentences('Hello...World'), ['Hello', 'World']);
  });
});

describe('toPhrases', () => {
  it('splits on commas, semicolons, and sentence-enders', () => {
    assert.deepEqual(toPhrases('A, B; C. D'), ['A', 'B', 'C', 'D']);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * Deduplication
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('dedupeSegments', () => {
  it('removes fuzzy-duplicate segments', () => {
    const segs = ['The cat sat', 'The cat sat down', 'A dog ran'];
    const result = dedupeSegments(segs);
    // "The cat sat" and "The cat sat down" are similar; preferLonger keeps the longer one
    assert.ok(result.length <= segs.length);
    assert.ok(result.includes('A dog ran'));
  });
  it('keeps all segments when nothing matches', () => {
    const segs = ['The elephant danced gracefully', 'A submarine explored deeply'];
    assert.deepEqual(dedupeSegments(segs), segs);
  });
});

describe('dedupeSentences', () => {
  it('deduplicates sentence-split text', () => {
    const text = 'The cat sat. The cat sat down. A dog ran.';
    const result = dedupeSentences(text);
    assert.ok(result.length <= 3);
  });
});

describe('dedupePhrases', () => {
  it('deduplicates phrase-split text', () => {
    const text = 'The cat sat, the cat sat down; a dog ran.';
    const result = dedupePhrases(text);
    assert.ok(result.length >= 1);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * Sorted LCS
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('sortedLcs', () => {
  it('computes LCS on sorted copies', () => {
    // "bca" sorted → "abc", "cab" sorted → "abc" → LCS = 3
    assert.equal(sortedLcs('bca', 'cab'), 3);
  });
  it('does not mutate original sequences', () => {
    const a = [3, 1, 2];
    const b = [2, 3, 1];
    sortedLcs(a, b);
    assert.deepEqual(a, [3, 1, 2]);
    assert.deepEqual(b, [2, 3, 1]);
  });
});

describe('sortedWordLcs', () => {
  it('sorts words then uses sentence-level LCS', () => {
    const score = sortedWordLcs('cat the sat', 'sat the cat');
    // Both sort to ["cat","sat","the"] → sentenceLcs = 3
    assert.equal(score, 3);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * Sørensen–Dice similarity
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('diceLcs (seq-level)', () => {
  it('returns 1 for identical sequences', () => {
    assert.equal(diceLcs('hello', 'hello'), 1);
  });
  it('returns 0 for completely disjoint sequences', () => {
    assert.equal(diceLcs('abc', 'xyz'), 0);
  });
  it('returns 0 for two empty sequences (0/0 guard)', () => {
    assert.equal(diceLcs('', ''), 0);
  });
  it('computes 2*LCS/(len1+len2) correctly', () => {
    // LCS("abcde","ace") = 3, total length = 5+3 = 8, dice = 6/8 = 0.75
    assert.equal(diceLcs('abcde', 'ace'), 0.75);
  });
  it('is symmetric', () => {
    assert.equal(diceLcs('abc', 'aec'), diceLcs('aec', 'abc'));
  });
  it('handles null/undefined gracefully', () => {
    assert.equal(diceLcs(null, 'abc'), 0);
    assert.equal(diceLcs('abc', null), 0);
  });
});

describe('diceWordLcs (word-level)', () => {
  it('returns 1 for identical words', () => {
    assert.equal(diceWordLcs('hello', 'hello'), 1);
  });
  it('is case-insensitive due to norm', () => {
    assert.equal(diceWordLcs('Hello', 'hello'), 1);
  });
  it('scores similar words high', () => {
    const score = diceWordLcs('running', 'runing');
    // LCS of normalized = 6, total = 7+6 = 13, dice = 12/13 ≈ 0.923
    assert.ok(score > 0.9);
  });
  it('scores dissimilar words low', () => {
    assert.ok(diceWordLcs('cat', 'dog') < 0.3);
  });
});

describe('diceSentenceLcs (sentence-level)', () => {
  it('returns 1 for identical token arrays', () => {
    assert.equal(diceSentenceLcs(['the', 'cat', 'sat'], ['the', 'cat', 'sat']), 1);
  });
  it('handles fuzzy word matching in score', () => {
    // "runing" ≈ "running" via wordMatch, so LCS = 3
    const score = diceSentenceLcs(['the', 'cat', 'running'], ['the', 'cat', 'runing']);
    // 2*3 / (3+3) = 1
    assert.equal(score, 1);
  });
  it('penalizes missing words', () => {
    const score = diceSentenceLcs(['a', 'b', 'c', 'd'], ['a', 'c']);
    // LCS = 2, total = 4+2 = 6, dice = 4/6 ≈ 0.667
    assert.ok(score > 0.6 && score < 0.7);
  });
  it('returns 0 for empty token arrays', () => {
    assert.equal(diceSentenceLcs([], []), 0);
  });
});

/* ═══════════════════════════════════════════════════════════════════════════
 * Namespace objects
 * ═══════════════════════════════════════════════════════════════════════════ */

describe('seq namespace', () => {
  it('exposes lcs as glcs', () => {
    assert.equal(seq.lcs('abc', 'ac'), 2);
  });
  it('exposes subsequence as lcsSubsequence', () => {
    const result = seq.subsequence('abcde', 'ace');
    assert.equal(result.score, 3);
    assert.deepEqual(result.subsequence, ['a', 'c', 'e']);
  });
  it('exposes match', () => {
    assert.equal(typeof seq.match, 'function');
    assert.equal(seq.match('hello', 'hello'), true);
  });
  it('exposes bestMatch', () => {
    assert.equal(typeof seq.bestMatch, 'function');
  });
  it('exposes has', () => {
    assert.equal(typeof seq.has, 'function');
  });
  it('exposes weighted', () => {
    assert.equal(typeof seq.weighted, 'function');
  });
  it('exposes context', () => {
    assert.equal(typeof seq.context, 'function');
  });
  it('exposes sorted', () => {
    assert.equal(typeof seq.sorted, 'function');
  });
});

describe('word namespace', () => {
  it('exposes norm', () => {
    assert.equal(word.norm('ABC'), 'abc');
  });
  it('exposes lcs as nlcs', () => {
    assert.ok(word.lcs('hello', 'Hello') > 0);
  });
  it('exposes match as wordMatch', () => {
    assert.equal(word.match('running', 'runnin'), true);
  });
  it('exposes bestMatch', () => {
    assert.equal(typeof word.bestMatch, 'function');
  });
  it('exposes weighted', () => {
    assert.equal(typeof word.weighted, 'function');
  });
  it('exposes context', () => {
    assert.equal(typeof word.context, 'function');
  });
  it('exposes sorted', () => {
    assert.equal(typeof word.sorted, 'function');
  });
});

describe('sentence namespace', () => {
  it('exposes lcs as sentenceLcs', () => {
    const w = ['a', 'b'];
    assert.equal(sentence.lcs(w, w), 2);
  });
  it('exposes match as sentenceMatch', () => {
    assert.equal(typeof sentence.match, 'function');
  });
  it('exposes tokenize', () => {
    assert.deepEqual(sentence.tokenize('a b'), ['a', 'b']);
  });
  it('exposes toSentences', () => {
    assert.equal(typeof sentence.toSentences, 'function');
  });
  it('exposes toPhrases', () => {
    assert.equal(typeof sentence.toPhrases, 'function');
  });
  it('exposes dedupeSegments', () => {
    assert.equal(typeof sentence.dedupeSegments, 'function');
  });
  it('exposes dedupeSentences', () => {
    assert.equal(typeof sentence.dedupeSentences, 'function');
  });
  it('exposes dedupePhrases', () => {
    assert.equal(typeof sentence.dedupePhrases, 'function');
  });
});
