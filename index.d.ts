/**
 * nlp-lcs — LCS-based sequence matching utilities for NLP pipelines.
 */

// ── Shared types ────────────────────────────────────────────────────────────

/** Any value with a numeric `.length` property (string, array, typed array, …). */
type Sequence = Iterable<any> & { length: number };

/** Element comparator — returns `true` when two elements should be considered equal. */
type Comparator = (a: any, b: any) => boolean;

/** LCS scoring function signature. */
type LcsFunction = (seq1: Sequence, seq2: Sequence) => number;

/** Threshold/matcher function signature. */
type MatcherFunction = (seq1: Sequence, seq2: Sequence, score: number) => boolean;

/** Minimum score — either a static number or a function that computes one. */
type MinScore = number | ((seq1: Sequence, seq2: Sequence) => number);

/** Result returned by `bestMatch` / `bestHas` / `bestWeighted` / `bestContext` style functions. */
interface BestMatchResult<T = any> {
  /** The best-scoring candidate from the list. */
  value: T;
  /** Whether the best candidate meets the threshold. */
  match: boolean;
  /** Raw LCS-derived score of the best candidate. */
  score: number;
}

/** Result returned by `lcsSubsequence`. */
interface SubsequenceResult<T = any> {
  /** Length of the longest common subsequence. */
  score: number;
  /** The actual matched elements. */
  subsequence: T[];
}

/** Options for `dedupeSegments`. */
interface DedupeOptions {
  /** Keep the longer of two matching segments. Defaults to `true`. */
  preferLonger?: boolean;
  /** Tokenizer function. Defaults to whitespace split. */
  tokenizer?: (str: string) => string[];
}

// ── LRU cache ───────────────────────────────────────────────────────────────

interface LruCache<K = string, V = any> {
  /** Retrieve a value, refreshing its recency. Returns `undefined` on miss. */
  get(key: K): V | undefined;
  /** Store a value, evicting the oldest entry if the cache is full. */
  set(key: K, value: V): V;
}

// ── Utility exports ─────────────────────────────────────────────────────────

/**
 * Get the length of any value.
 * Tries `.length`, then `.size()`, then `parseInt` as a numeric fallback.
 * Returns 0 when the value is null, undefined, or has no recognizable length.
 */
export declare function len(x: any): number;

/** Default strict-equality comparator. */
export declare const defaultCompare: Comparator;

/**
 * Create a simple LRU cache backed by insertion-ordered `Map`.
 * @param limit Maximum number of entries. Defaults to `4096`.
 */
export declare function createLruCache<K = string, V = any>(limit?: number): LruCache<K, V>;

/** Check whether a value is a string primitive or `String` object. */
export declare function isString(x: any): x is string;

/**
 * Convert any value to a string.
 * Strings pass through; objects are JSON-stringified; circular refs fall back to `String()`.
 */
export declare function stringify(x: any): string;

/** Build an order-independent cache key from two sequences. */
export declare function makeMemoKey(seq1: any, seq2: any): string;

/**
 * Pick the narrowest typed-array constructor whose element range covers `maxValue`.
 * Falls back to `Array` for values beyond `Uint32` range.
 */
export declare function selectArrayType(
  maxValue: number,
): typeof Uint8Array | typeof Uint16Array | typeof Uint32Array | typeof Array;

// ── seq namespace — general / arbitrary-element sequences ───────────────────

export declare const seq: {
  /**
   * General LCS score between any two iterables.
   * Pluggable comparator enables the recursive hierarchy.
   * Optional `minScore` enables early termination when a threshold can't be reached.
   */
  lcs(
    seq1: Sequence,
    seq2: Sequence,
    compare?: Comparator,
    minScore?: MinScore,
  ): number;

  /**
   * LCS with backtracking — returns both the score and the actual matched elements.
   */
  subsequence<T = any>(
    seq1: Iterable<T> & { length: number },
    seq2: Iterable<T> & { length: number },
    compare?: Comparator,
  ): SubsequenceResult<T>;

  /** Test whether `score >= floor(0.8 × max(len1, len2))`. */
  scoreMatch(seq1: Sequence, seq2: Sequence, score?: number): boolean;

  /** `true` if LCS score ≥ 80 % of the longer sequence's length. */
  match(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): boolean;

  /** Find the highest-scoring candidate from a list. */
  bestMatch<T extends Sequence>(
    query: Sequence,
    candidates: T[],
    lcs?: LcsFunction,
    matcher?: MatcherFunction,
  ): BestMatchResult<T>;

  /** Test whether `score >= floor(0.8 × min(len1, len2))`. */
  scoreHas(seq1: Sequence, seq2: Sequence, score?: number): boolean;

  /** `true` if LCS score ≥ 80 % of the shorter sequence's length (containment). */
  has(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): boolean;

  /** Best containment match from a list. */
  bestHas<T extends Sequence>(
    query: Sequence,
    candidates: T[],
    lcs?: LcsFunction,
  ): BestMatchResult<T>;

  /** LCS score × (minLen / maxLen) — penalizes length disparity. */
  weighted(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): number;

  /** Best weighted match from a list. */
  bestWeighted<T extends Sequence>(query: Sequence, candidates: T[]): BestMatchResult<T>;

  /** LCS score + (maxLen / minLen) — rewards longer containing sequences. */
  context(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): number;

  /** Best context match from a list. */
  bestContext<T extends Sequence>(query: Sequence, candidates: T[]): BestMatchResult<T>;

  /** LCS on sorted copies — order-insensitive comparison. */
  sorted(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): number;

  /** Sørensen–Dice similarity: 2×LCS / (len1+len2). Returns a value in [0, 1]. */
  dice(seq1: Sequence, seq2: Sequence, lcs?: LcsFunction): number;
};

// ── word namespace — character-level (single-word) comparisons ──────────────

export declare const word: {
  /** Normalize a value to a trimmed, NFD-decomposed, lowercased string. */
  norm(x: any): string;

  /** Normalized character-level LCS score. */
  lcs(a: string, b: string): number;

  /** Fuzzy word match (80 % character-level threshold). */
  match(a: string, b: string): boolean;

  /** Best fuzzy word from a list. */
  bestMatch(query: string, candidates: string[]): BestMatchResult<string>;

  /** Weighted character-level score. */
  weighted(a: string, b: string): number;

  /** Best weighted word from a list. */
  bestWeighted(query: string, candidates: string[]): BestMatchResult<string>;

  /** Context-rewarding character-level score. */
  context(a: string, b: string): number;

  /** Best context word from a list. */
  bestContext(query: string, candidates: string[]): BestMatchResult<string>;

  /** Sorted word-level LCS (tokenizes, sorts, then sentence LCS). */
  sorted(a: string, b: string): number;

  /** Sørensen–Dice similarity at the character level. Returns a value in [0, 1]. */
  dice(a: string, b: string): number;
};

// ── sentence namespace — tokenized-sentence (word-sequence) comparisons ─────

export declare const sentence: {
  /** Word-level fuzzy LCS score between two token arrays. */
  lcs(words1: string[], words2: string[]): number;

  /** `true` if the sentences meet the 80 % word-structure threshold. */
  match(words1: string[], words2: string[]): boolean;

  /** Best matching sentence from a list. */
  bestMatch(query: string[], candidates: string[][]): BestMatchResult<string[]>;

  /** Split a string on whitespace. */
  tokenize(str: string): string[];

  /** Split text on sentence-ending punctuation (`.`, `!`, `?`). */
  toSentences(str: string): string[];

  /** Split text on phrase-level punctuation (`.`, `!`, `?`, `,`, `;`). */
  toPhrases(str: string): string[];

  /** Remove fuzzy-duplicate text segments. */
  dedupeSegments(segments: string[], options?: DedupeOptions): string[];

  /** Split text into sentences and dedupe. */
  dedupeSentences(text: string, options?: DedupeOptions): string[];

  /** Split text into phrases and dedupe. */
  dedupePhrases(text: string, options?: DedupeOptions): string[];

  /** Sørensen–Dice similarity at the sentence level. Returns a value in [0, 1]. */
  dice(words1: string[], words2: string[]): number;
};
