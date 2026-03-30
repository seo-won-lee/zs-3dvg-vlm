# --------------------------------------------------------
# Computes McNemar's test to compare baseline and modified models
# Evaluates statistical significance of performance differences
# --------------------------------------------------------

import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def get_results(pred_dir):
    """
    Compute acc@25, acc@50, and uniqueness for each sample.
    
    """

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    pred_files.sort()

    acc25 = []
    acc50 = []
    is_unique = []

    for pred_file in pred_files:

        pred_file = os.path.join(pred_dir, pred_file)
        preds = load_json(pred_file)

        for pred_entry in preds:

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']

            # ScanRefer annotation에 보통 있음
            unique_flag = pred_entry.get('unique', False)

            iou = calc_iou(gt_bbox, pred_bbox)

            acc25.append(1 if iou >= 0.25 else 0)
            acc50.append(1 if iou >= 0.5 else 0)
            is_unique.append(unique_flag)

    return {
        "acc25": np.array(acc25),
        "acc50": np.array(acc50),
        "unique": np.array(is_unique)
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
    # print("Contingency Table:", table)
    print(f"(p={result.pvalue:.3f})")

    if result.pvalue < 0.05:
        print("Statistically significant")
    else:
        print("Not statistically significant")


def run_mcnemar(baseline_dir, ours_dir):

    baseline = get_results(baseline_dir)
    ours = get_results(ours_dir)


    # unique mask
    unique_mask = baseline["unique"] == True

    compute_mcnemar(
        baseline["acc25"][unique_mask],
        ours["acc25"][unique_mask],
        "Unique@25"
    )

    compute_mcnemar(
        baseline["acc50"][unique_mask],
        ours["acc50"][unique_mask],
        "Unique@50"
    )

    # multiple mask
    multiple_mask = baseline["unique"] == False

    compute_mcnemar(
        baseline["acc25"][multiple_mask],
        ours["acc25"][multiple_mask],
        "Multiple@25"
    )

    compute_mcnemar(
        baseline["acc50"][multiple_mask],
        ours["acc50"][multiple_mask],
        "Multiple@50"
    )
    
    # Overall
    compute_mcnemar(baseline["acc25"], ours["acc25"], "Acc@25")
    compute_mcnemar(baseline["acc50"], ours["acc50"], "Acc@50")


if __name__ == "__main__":

    baseline_dir = './outputs/qwen2-vl-7b/scanrefer/val/pred'

    ours_dir = './outputs/qwen2-vl-7b/scanrefer/modified/pred'

    run_mcnemar(baseline_dir, ours_dir)