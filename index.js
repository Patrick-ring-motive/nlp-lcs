const lcsMemo = new Map();

const glcs = function generalLongestCommonSubsequence(seq1, seq2, compare = (x, y) => (x === y))) {
    "use strict";
    const key = String([seq1, seq2].sort());
    if (lcsMemo.has(key)) {
        return lcsMemo.get(key);
    }
    let array1 = [...seq1 ?? []];
    let array2 = [...seq2 ?? []];
    if (array2.length > array1.length) {
        [array1, array2] = [array2, array1];
    }
    const [arr1, arr2] = [array1, array2];
    const arr1_length = arr1.length;
    const arr2_length = arr2.length;
    const dp = Array(arr1_length + 1).fill(0).map(() => new Uint8Array(arr2_length + 1));
    const dp_length = dp.length;
    for (let i = 1; i !== dp_length; ++i) {
        const dpi_length = dp[i].length;
        for (let x = 1; x !== dp_length; ++x) {
            if (arr1[i - 1] === arr2[x - 1]) {
                dp[i][x] = dp[i - 1][x - 1] + 1
            } else {
                dp[i][x] = Math.max(dp[i][x - 1], dp[i - 1][x])
            }
        }
    }
    const score = dp[arr1_length][arr2_length];
    lcsMemo.set(key, score);
    return score;
};

const norm = x => String(x).normalize('NFD').toLowerCase();

const nlcs = function normalizedLongestCommonSubsequence(str1, str2) {
    return glcs(norm(str1), norm(str2));
};

const lcsMatch = (seq1, seq2, lcs = glcs) => {
    return lcs(seq1, seq2) >= Math.floor(0.8 * Math.max(seq1.length, seq2.length));
};

const lcsHas = (seq1, seq2, lcs = glcs) => {
    return lcs(seq1, seq2) >= Math.floor(0.8 * Math.min(seq1.length, seq2.length));
};

const wordMatch = (seq1, seq2) => {
    return lcsMatch(seq1, seq2, nlcs);
};

const sentenceLcs = (words1, words2) => {
    return glcs(words1, words2, wordMatch);
};

const sentenceMatch = (seq1, seq2) => {
    return lcsMatch(seq1, seq2, sentenceLcs);
};

const weightedLcs = (seq1, seq2, lcs = glcs) => {
    return lcs(seq1, seq2) * Math.min(seq1.length, seq2.length) / Math.max(seq1.length, seq2.length, 1);
};

const contextLcs = (seq1, seq2, lcs = glcs) => {
    return lcs(seq1, seq2) + Math.max(seq1.length, seq2.length) / (Math.min(seq1.length, seq2.length) || 1);
};
