# SCORE-Bench corrected ground truth

Files here named `<original-gt-name>__correct_truth.txt` override the stock
`SCORE-Bench/content-gt/` file for that case (see `iter_score_bench_cases` in
`scripts/score_unstructured_bench.py`).

## Rules for adding a correction

1. **The page image is the arbiter, never engine output.** Render the page and
   verify the defect against pixels before writing anything. Corrections
   derived from our extraction would grade the engine against itself.
2. **Verify with the scorer's own tokenizer** (`clean_score_bench_text` +
   `tokenize`), not ad-hoc regex counts. Two of the first three candidate
   "GT defects" turned out to be analysis bugs: one was our own line
   duplication (fixed in the engine instead), the other was counting the
   dashes of the `--- Unstructured ... ---` marker lines that the cleaner
   strips.
3. **Minimal edits.** Start from the stock GT, change only what the image
   contradicts, and keep the Unstructured plaintext format so tokenization
   behaves identically.
4. **Document the defect in-file** using a pseudo-marker line
   (`---...--- Unstructured Correct-Truth Note: ...`) — the cleaner strips
   any line of 10+ dashes followed by `Unstructured `, so notes never leak
   into scoring.

## Current corrections

- `Tobacco-Lab-Reproducibility-Tables-p001...` — removed the
  `Source: https://www.industrydocuments.ucsf.edu/...` page-footer block:
  UCSF archive provenance metadata, not printed on the page (verified against
  a 3x render of the page bottom, which shows only the ATX02 0235774 stamp).
