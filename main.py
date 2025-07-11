import argparse
from os import makedirs
from os.path import join

from src.conf import DATA_SYNTHESIS_FILE_NAME, RESULTS_DIR
from src.eval.explanability import get_breakdown_per_contexts
from src.eval.metrics import build_metrics_from_confusion, get_confusion_matrix
from src.mistral.inference.batch import run_batch_mistral, upload_file
from src.mistral.prep import (
    generate_bin_cls_request_file,
    generate_multi_cls_request_file,
)
from src.mistral.prep._utils import get_task_full_name
from src.mistral.results._binary_cls import get_binary_cls_result
from src.mistral.results._multi_cls import get_multi_cls_result
from src.processing.ipcc.parse import load_and_process_ipcc_reports
from src.processing.ipcc.request_gen import (
    generate_request_file_for_accurate_sample_gen,
)
from src.processing.quota_climat import load_quota_climat_dataset
from src.processing.task import (
    build_bin_cls_dataset,
    build_finetune_dataset,
    build_multi_cls_dataset,
)
from src.utils import logger


def load():
    load_quota_climat_dataset()
    load_and_process_ipcc_reports()


def generate_synthetic_samples():
    text_blocks = load_and_process_ipcc_reports()

    generate_request_file_for_accurate_sample_gen(ipcc_report_blocks=text_blocks)
    upload_file(DATA_SYNTHESIS_FILE_NAME)
    run_batch_mistral(
        file_name=DATA_SYNTHESIS_FILE_NAME,
        model="mistral-large-latest",
        job_type="data-synthesis",
    )


def build(args):
    task_full_name = get_task_full_name(args.task, args.eval_set, args.few_shot)
    logger.info("Building dataset for %s", task_full_name)
    if args.task == "binary":
        build_bin_cls_dataset()
        generate_bin_cls_request_file(task_full_name)
    if args.task == "multiclass":
        if args.eval_set == "global":
            build_multi_cls_dataset()
        else:
            build_finetune_dataset()

        generate_multi_cls_request_file(
            file_name=task_full_name,
            dataset=args.eval_set,
            examples=int(args.few_shot),  # One example per class
        )
    logger.error("Unknown task: %s", args.task)


def run(args):
    task_full_name = get_task_full_name(args.task, args.eval_set, args.few_shot)
    upload_file(task_full_name)

    run_batch_mistral(
        file_name=task_full_name,
        model=args.model,
        job_type="multi-cls" if args.task == "multiclass" else "binary-cls",
    )


def evaluate(args):
    task_full_name = get_task_full_name(args.task, args.eval_set, args.few_shot)

    result_sub_dir = join(RESULTS_DIR, task_full_name)
    makedirs(result_sub_dir, exist_ok=True)

    if args.task == "binary":
        result = get_binary_cls_result(model=args.model)
    else:
        result = get_multi_cls_result(model=args.model, task=task_full_name)

    confusion_matrix = get_confusion_matrix(result)
    metrics = build_metrics_from_confusion(confusion_matrix)

    metrics.to_csv(join(result_sub_dir, "metrics.csv"))
    confusion_matrix.to_csv(join(result_sub_dir, "confusion_matrix.csv"))
    result.to_csv(join(result_sub_dir, "prediction.csv"), index=False)

    # A bit of explanability in case of binary classification
    if args.task == "binary":
        (
            accurate_fn_per_personae,
            accurate_fn_per_emotion,
            misinformation_fn_per_sub_class,
        ) = get_breakdown_per_contexts(result)
        accurate_fn_per_personae.to_csv(
            join(result_sub_dir, "accurate_label_false_negatives_per_personae.csv")
        )
        accurate_fn_per_emotion.to_csv(
            join(result_sub_dir, "accurate_label_false_negatives_per_emotion.csv")
        )
        misinformation_fn_per_sub_class.to_csv(
            join(
                result_sub_dir, "misinformation_label_false_negatives_per_sub_class.csv"
            )
        )

    logger.info("Results were saved to %s", result_sub_dir)


def classify(args):
    logger.info(
        "Classification task: %s, stage: %s, model: %s",
        args.task,
        args.stage,
        args.model,
    )
    logger.info("Few-shot mode: %s", args.few_shot)
    logger.info("Evaluation set: %s", args.eval_set)

    if args.task == "binary" and not args.few_shot:
        logger.error("Error: few-shot cannot be False when task is 'binary'")
        return

    if args.stage == "build":
        build(args)
    if args.stage == "run":
        run(args)
    if args.stage == "eval":
        evaluate(args)

    logger.info("All done.")


def main():
    parser = argparse.ArgumentParser(
        description="Document processing and classification tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Load command
    load_parser = subparsers.add_parser(
        "load", help="Load data. Warning, uses Mitsral's API."
    )
    load_parser.set_defaults(func=load)

    # Generate synthetic accurate quote samples
    synth_parser = subparsers.add_parser(
        "synthesize", help="Generate synthetic accurate quote samples."
    )
    synth_parser.set_defaults(func=generate_synthetic_samples)

    # Classify command
    classify_parser = subparsers.add_parser("class", help="Classication task")
    classify_parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        required=True,
        help="Classification task type",
    )
    classify_parser.add_argument(
        "--stage",
        choices=["build", "run", "eval"],
        required=True,
        help=(
            "Processing stage.\n"
            "Build: Creates all necessary files "
            "from processed data (need for the OCR job to be finished).\n"
            "Run: Upload the files to Mistral api and launch inference job.\n"
            "Eval: Evaluate the results of the inference job and save metrics."
        ),
    )
    classify_parser.add_argument(
        "--model",
        choices=[
            "ministral-3b-latest",
            "ministral-8b-latest",
            "mistral-small-latest",
            # "ministral-3b-latest-v0", fine-tuned model
        ],
        default="ministral-3b-latest",
        help="Model to use (default: ministral-3b-latest)",
    )
    classify_parser.add_argument(
        "--few-shot",
        action="store_true",
        default=True,
        help=(
            "Use few-shot learning (default: True).\n"
            "Zero Shot is not available for binary classification"
        ),
    )
    classify_parser.add_argument(
        "--eval-set",
        choices=["global", "validation"],
        default="global",
        help=(
            "Evaluation set to use.\n"
            "'global' for all data\n"
            "'validation' for validation set used for fine-tuning"
        ),
    )
    classify_parser.set_defaults(func=classify)

    args = parser.parse_args()
    args.func()


if __name__ == "__main__":
    main()
