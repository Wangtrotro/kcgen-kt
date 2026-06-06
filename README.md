# Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems
This repository contains the code for the paper <a href="https://arxiv.org/abs/2502.18632">Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems</a>. The primary contributions here include 1. A completely automatic KC generation pipeline using solution AST. 2. Utilizing KC mastery level for knowledge tracing task 

## Setup

### Data
We use [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) dataset. The dataset can be downloaded as follow:
```
pip install gdown
cd data
bash data.sh
```

### Environment
We use Python 3.8 in the development of this work. Run the following to set up a Conda environment and install the packages required. After activate the conda environment, we install pytorch with cuda version 11.8, you may want to install according to your own situation. More info can be found on the PyTorch website: https://pytorch.org/get-started/locally/ :
```
conda create --name <env_name> python=3.8
conda activate <env_name>
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 
conda env update --name <env_name> --file environment.yml
```

---

## Path-aware Knowledge Tracing (Extension)

This extension validates whether **solution path identity improves programming knowledge tracing and exercise recommendation**. The core hypothesis: students who solve the same problem using different strategies activate different KC subsets, and recognizing this improves cognitive state estimation.

### Quick Start

```bash
# Step 0: Setup base environment + download data (see above)

# Step 1: Discover solution paths (clusters correct submissions per problem)
python path_discovery.py --n_paths 3

# Step 2: Generate path-level KCs (requires OPENAI_API_KEY, or use --fallback)
python path_kc_gen.py --fallback  # heuristic, no API needed
# OR
python path_kc_gen.py             # uses GPT-4o for path-specific KC generation

# Step 3: Preliminary validation (no GPU needed)
python analyze_path_difference.py

# Step 4: Run the ablation experiment (requires GPU)
python run_ablation.py --mode all --epochs 3

# Or run individual experiments:
python main_path_kt.py path_mode=hard                        # real paths
python main_path_kt.py path_mode=hard randomize_paths=true   # random paths (ablation)
python main_path_kt.py path_mode=none                        # no paths (baseline)
```

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Path Discovery (path_discovery.py)                         │
│   Student correct submissions → GraphCodeBERT embedding             │
│   → Agglomerative Clustering → Path prototypes + assignments        │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Path-level KC Generation (path_kc_gen.py)                  │
│   Per-path representative codes → GPT-4o → Path-specific KCs        │
│   → Path-conditioned activation matrix (problem × path → KC subset) │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Path-aware KT (main_path_kt.py)                           │
│   Student submission → Path recognizer → KC activation mask          │
│   → LSTM mastery (path-weighted) → Llama-3 code generation           │
│   → Multi-task loss (generation + KC mastery + correctness)          │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 4: Ablation (run_ablation.py)                                 │
│   Compare: real_path vs random_path vs no_path                       │
│   Key test: if real > random → path identity carries signal          │
└─────────────────────────────────────────────────────────────────────┘
```

### New Files

| File | Purpose |
|------|---------|
| `path_discovery.py` | Cluster solutions per problem, assign path IDs to all submissions |
| `path_kc_gen.py` | Generate path-specific KCs using LLM (or heuristic fallback) |
| `path_data_loader.py` | Extended data loader with path information |
| `path_trainer.py` | Path-conditioned KC activation in the training loop |
| `main_path_kt.py` | Main experiment entry point for path-aware KT |
| `configs_path_kt.yaml` | Configuration for path-aware experiments |
| `run_ablation.py` | Automated ablation: real-path vs random-path vs no-path |
| `analyze_path_difference.py` | Statistical validation of the path premise |
| `RESEARCH_ANALYSIS.md` | Detailed research positioning analysis |

### Key Experiment: Real Path vs Random Path

The most important ablation keeps the model architecture and KC set identical, and only shuffles path assignments within each problem. This controls for:
- Extra parameters (same model capacity)
- KC granularity (same KC set)
- Path count (same number of clusters)

If `real_path` outperforms `random_path`, the gain is attributable to path identity itself — not to having more features or parameters.

