# --------------------------------------------------------
# Computes McNemar's test to compare baseline and modified models
# Evaluates statistical significance of performance differences
# --------------------------------------------------------

import json
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def get_results(pred_dir):
    """
    Compute acc@25, acc@50, and whether each sample is easy or view-dependent.
    
    """

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    pred_files.sort()

    acc25 = []
    acc50 = []
    easy = []
    view_dep = []

    for pred_file in pred_files:

        pred_file = os.path.join(pred_dir, pred_file)

        if not os.path.exists(pred_file):
            continue

        preds = load_json(pred_file)

        for pred_entry in preds:

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']

            iou = calc_iou(gt_bbox, pred_bbox)

            acc25.append(1 if iou >= 0.25 else 0)
            acc50.append(1 if iou >= 0.5 else 0)

            easy.append(pred_entry['easy'])
            view_dep.append(pred_entry['view_dep'])

    return {
        "acc25": np.array(acc25),
        "acc50": np.array(acc50),
        "easy": np.array(easy),
        "view_dep": np.array(view_dep)
    }


def compute_mcnemar(a, b, name):
    """
    Perform McNemar's test to evaluate statistical signicance 
    between baseline and modified model results.
    
    """

    assert len(a) == len(b)

    both_correct = np.sum((a == 1) & (b == 1))
    a_only = np.sum((a == 1) & (b == 0))
    b_only = np.sum((a == 0) & (b == 1))
    both_wrong = np.sum((a == 0) & (b == 0))

    table = [[both_correct, a_only],
             [b_only, both_wrong]]

    result = mcnemar(table, exact=True)

    print(f"\n{name}")
    print(f"(p={result.pvalue:.3f})")

    if result.pvalue < 0.05:
        print("Statistically significant")
    else:
        print("Not statistically significant")


def run_mcnemar(baseline_dir, ours_dir):

    baseline = get_results(baseline_dir)
    ours = get_results(ours_dir)

    print("Baseline samples:", len(baseline["acc25"]))
    print("Ours samples:", len(ours["acc25"]))

    assert len(baseline["acc25"]) == len(ours["acc25"]), \
        "Prediction counts do not match!"

    acc25_base = baseline["acc25"]
    acc25_ours = ours["acc25"]

    acc50_base = baseline["acc50"]
    acc50_ours = ours["acc50"]

    easy_mask = baseline["easy"]
    view_dep_mask = baseline["view_dep"]

    # Easy
    compute_mcnemar(
        acc25_base[easy_mask],
        acc25_ours[easy_mask],
        "Easy"
    )

    # Hard
    compute_mcnemar(
        acc25_base[~easy_mask],
        acc25_ours[~easy_mask],
        "Hard"
    )

    # View Dependent
    compute_mcnemar(
        acc25_base[view_dep_mask],
        acc25_ours[view_dep_mask],
        "View-Dep"
    )

    # View Independent
    compute_mcnemar(
        acc25_base[~view_dep_mask],
        acc25_ours[~view_dep_mask],
        "View-Indep"
    )

    # Overall
    compute_mcnemar(acc25_base, acc25_ours, "Acc@25")
    compute_mcnemar(acc50_base, acc50_ours, "Acc@50")


if __name__ == "__main__":
    
    baseline_dir = './outputs/qwen2-vl-7b/nr3d/val/pred'

    ours_dir = './outputs/qwen2-vl-7b/nr3d/modified/pred'

    run_mcnemar(baseline_dir, ours_dir)