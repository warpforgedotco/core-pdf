# OCR word lists

This directory vendors full compressed upstream word-frequency data used for
pure-Python OCR text repair.

## Files

- `norvig_count_1w.txt.gz`
  - Source: https://github.com/BartMassey/wordlists
  - Upstream file: `count_1w.txt.gz`
  - Data origin: Peter Norvig's 1/3M frequent English word counts derived from
    the Google Web Trillion Word Corpus.
  - License notice: see `LICENSE_BartMassey_wordlists.txt` and
    `README_BartMassey_count_1w.md`.

- `wordninja_words.txt.gz`
  - Source: https://github.com/keredson/wordninja
  - Upstream file: `wordninja/wordninja_words.txt.gz`
  - License notice: see `LICENSE_wordninja.txt`.

The compressed upstream files are intentionally stored whole so repair
heuristics can choose their own cutoffs without adding a third-party runtime
dependency.
