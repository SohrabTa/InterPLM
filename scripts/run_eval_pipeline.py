import argparse
import sys
from pathlib import Path
from interplm.analysis.concepts.compare_activations import analyze_all_shards_in_set
from interplm.analysis.concepts.calculate_f1 import combine_metrics_across_shards
from interplm.analysis.concepts.report_metrics import report_metrics

# Add the project root to sys.path to allow importing interplm
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run full InterPLM evaluation pipeline"
    )
    parser.add_argument(
        "--sae_dir",
        type=str,
        required=True,
        help="Path to SAE directory (containing ae_normalized.pt)",
    )
    parser.add_argument(
        "--aa_embds_dir",
        type=str,
        required=True,
        help="Path to directory containing embedding shards",
    )
    parser.add_argument(
        "--eval_data_root",
        type=str,
        required=True,
        help="Path to processed annotations directory (containing valid/ and test/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/eval_pipeline",
        help="Root directory for storing evaluation results",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0, 0.15, 0.5, 0.6, 0.8],
        help="List of thresholds for activation comparison",
    )
    parser.add_argument(
        "--top_threshold",
        type=float,
        default=0.5,
        help="F1 threshold for reporting top pairings",
    )
    parser.add_argument(
        "--is_sparse",
        action="store_true",
        default=True,
        help="Whether to use sparse operations (recommended)",
    )

    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    aa_embds_dir = Path(args.aa_embds_dir)
    eval_data_root = Path(args.eval_data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. & 2. Run for both valid and test sets
    eval_sets = ["valid", "test"]
    f1_paths = {}

    for eval_set in eval_sets:
        print(f"\n--- Processing {eval_set} set ---")
        eval_set_dir = eval_data_root / eval_set
        counts_dir = output_root / f"{eval_set}_counts"

        # Step 1: Compare activations
        print(f"Step 1: Comparing activations for {eval_set}...")
        analyze_all_shards_in_set(
            sae_dir=sae_dir,
            aa_embds_dir=aa_embds_dir,
            eval_set_dir=eval_set_dir,
            output_dir=counts_dir,
            threshold_percents=args.thresholds,
            is_sparse=args.is_sparse,
        )

        # Step 2: Combine metrics and calculate F1
        print(f"Step 2: Calculating F1 scores for {eval_set}...")
        f1_path = counts_dir / "concept_f1_scores.csv"
        combine_metrics_across_shards(
            eval_res_dir=counts_dir,
            eval_set_dir=eval_set_dir,
            threshold_percents=args.thresholds,
            custom_output_path=f1_path,
        )
        f1_paths[eval_set] = f1_path

    # Step 3: Report final metrics
    print("\n--- Final Step: Reporting metrics ---")
    report_metrics(
        valid_path=f1_paths["valid"],
        test_path=f1_paths["test"],
        top_threshold=args.top_threshold,
    )
    print("\nEvaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
