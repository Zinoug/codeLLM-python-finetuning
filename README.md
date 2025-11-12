# CodeT5+ Multi-Task Fine-Tuning Project

Fine-tuning CodeT5+ for code repair, bug detection, code search, and code summarization tasks.

## Project Structure

```
projectSEGA/
├── data-mining/              # Scripts to mine bug fixes from GitHub
├── data/                     # Mined datasets
└── models/                   # Fine-tuned model checkpoints
```

## Data Mining

Extract real bug fixes from GitHub repositories:

```bash
cd data-mining
pip install -r requirements.txt
python mine_top_level_bug_fixes.py
```

**Target repositories:** TheAlgorithms/Python, more-itertools, toolz, etc.

## Tasks

### 1. Code Repair
- **Input:** `fix a bug:\n{buggy_code}`
- **Output:** Fixed code
- **Metric:** Pass@1

### 2. Bug Detection
- **Input:** `classify:\n{code}`
- **Output:** BUGGY or CORRECT
- **Metric:** Accuracy, F1-Score

### 3. Code Search
- **Input:** `search code:\n{query}\nChoices:...`
- **Output:** Answer index
- **Metric:** Accuracy

### 4. Code Summary
- **Input:** `summarize code:\n{code}`
- **Output:** Natural language description
- **Metric:** BLEU, ROUGE-L


## Dataset Status

- ✅ Code Repair: 36 validated samples (target: 500-1000)
- 🔄 Code Search/Summary: 0/400-500 function-docstring pairs
- 🔄 Bug Detection: Same as Code Repair dataset


## License

Academic project - Seoul National University of Technology and Science
