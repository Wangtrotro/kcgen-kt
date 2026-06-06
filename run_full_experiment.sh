#!/bin/bash
# ============================================================
#  Full Mid-term Experiment: Path-aware KT on Apple Silicon
#  Run this on your M4 Pro MacBook
# ============================================================
#
#  Prerequisites:
#    conda activate <env>  (or your venv)
#    pip install torch transformers peft accelerate
#    pip install pandas numpy scikit-learn scipy tqdm
#    pip install openai sentence-transformers
#    cd data && bash data.sh  (download dataset)
#
#  Memory guide:
#    - 18GB M4 Pro: use batch_size=1, lora_r=64
#    - 36GB M4 Pro: use batch_size=2, lora_r=128
#
# ============================================================

set -e

echo "============================================================"
echo "  Path-aware KT: Full Mid-term Experiment"
echo "  $(date)"
echo "============================================================"

# --- Step 1: Path Discovery (if not already done) ---
if [ ! -f "path_cache/path_assignments.json" ]; then
    echo ""
    echo ">>> Step 1: Discovering solution paths..."
    python path_discovery.py --n_paths 3
else
    echo ">>> Step 1: Path discovery already done, skipping."
fi

# --- Step 2: Path KC Generation (if not done) ---
if [ ! -f "path_cache/path_kc_map.json" ] || [ $(python -c "import json; d=json.load(open('path_cache/path_kc_map.json')); print(len(d))") -lt 40 ]; then
    echo ""
    echo ">>> Step 2: Generating path-level KCs..."
    if [ -n "$OPENAI_API_KEY" ]; then
        python path_kc_gen.py --path_dir path_cache --model gpt-4o-mini
    else
        echo "    No OPENAI_API_KEY, using fallback heuristic"
        python path_kc_gen.py --path_dir path_cache --fallback
    fi
else
    echo ">>> Step 2: Path KC generation already done, skipping."
fi

# --- Step 3: Statistical Validation ---
echo ""
echo ">>> Step 3: Statistical validation of path premise..."
python analyze_path_difference.py --path_dir path_cache

# --- Step 4: Main Experiments (3 conditions) ---
echo ""
echo "============================================================"
echo "  Running Main KT Experiments (3 conditions × 5 epochs)"
echo "============================================================"

EPOCHS=5
CONFIG="configs_path_kt_apple.yaml"

# Experiment A: Baseline (no path conditioning)
echo ""
echo ">>> Experiment A: BASELINE (no path)"
python main_path_kt_apple.py --config $CONFIG --no_path --epochs $EPOCHS --seed 0

# Experiment B: Real paths
echo ""
echo ">>> Experiment B: REAL PATH"
python main_path_kt_apple.py --config $CONFIG --epochs $EPOCHS --seed 0

# Experiment C: Random paths (ablation)
echo ""
echo ">>> Experiment C: RANDOM PATH (ablation)"
python main_path_kt_apple.py --config $CONFIG --randomize --epochs $EPOCHS --seed 0

# --- Step 5: Multiple seeds for significance ---
echo ""
echo "============================================================"
echo "  Running additional seeds for statistical significance"
echo "============================================================"

for SEED in 1 2 3 4; do
    echo ""
    echo "--- Seed $SEED ---"
    python main_path_kt_apple.py --config $CONFIG --no_path --epochs $EPOCHS --seed $SEED
    python main_path_kt_apple.py --config $CONFIG --epochs $EPOCHS --seed $SEED
    python main_path_kt_apple.py --config $CONFIG --randomize --epochs $EPOCHS --seed $SEED
done

# --- Step 6: Collect and Compare Results ---
echo ""
echo "============================================================"
echo "  Collecting Results"
echo "============================================================"
python -c "
import json, os, numpy as np
from glob import glob

results = {'baseline': [], 'real_path': [], 'random_path': []}
for f in glob('checkpoints/results_*.json'):
    with open(f) as fh:
        r = json.load(fh)
    mode = r['mode']
    if 'RANDOM' in mode:
        results['random_path'].append(r['test_metrics'])
    elif 'none' in mode:
        results['baseline'].append(r['test_metrics'])
    else:
        results['real_path'].append(r['test_metrics'])

print()
print('=' * 60)
print('  FINAL COMPARISON')
print('=' * 60)
for mode in ['baseline', 'real_path', 'random_path']:
    if results[mode]:
        aucs = [float(r.get('auc', r.get('AUC', 0))) for r in results[mode] if 'auc' in r or 'AUC' in r]
        accs = [float(r.get('acc', r.get('ACC', 0))) for r in results[mode] if 'acc' in r or 'ACC' in r]
        if aucs:
            print(f'  {mode:15s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.3f}, ACC={np.mean(accs):.4f}±{np.std(accs):.3f} (n={len(aucs)})')
    else:
        print(f'  {mode:15s}: No results')
print('=' * 60)
"

echo ""
echo "Done! Check checkpoints/ for detailed results."
