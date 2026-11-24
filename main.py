# main entry point for cli execution

import os
import sys
import argparse


# to parse command-line arguments and route subcommands
def main():
    parser = argparse.ArgumentParser(
        description="FakeLens: Multi-Domain AI Image Forgery Detection & Attribution"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # server subcommand
    serve_parser = subparsers.add_parser("serve", help="Run the FakeLens web application server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    eval_parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact", "cifake"], required=True)
    eval_parser.add_argument("--task", type=str, choices=["binary", "multiclass"], required=True)
    eval_parser.add_argument("--model", type=str, required=True, help="Model ID (e.g. efficientnet_b0, cnn, random_forest, mlp, knn)")
    eval_parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    eval_parser.add_argument("--allow_test_split", action="store_true", help="Explicit authorization for protected test set")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Inspect or train a model family")
    train_parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact", "cifake"], required=True)
    train_parser.add_argument("--task", type=str, choices=["binary", "multiclass"], required=True)
    train_parser.add_argument("--model", type=str, choices=["efficientnet_b0", "cnn", "random_forest", "mlp", "knn"], required=True)
    train_parser.add_argument("--dry_run", action="store_true", help="Validate setup without gradient updates")
    train_parser.add_argument("--epochs", type=int, default=20)

    # preprocess subcommand
    prep_parser = subparsers.add_parser("preprocess", help="Run image or tabular preprocessing")
    prep_parser.add_argument("--image", type=str, required=True, help="Path to input image")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        from backend.api.server import run_server
        print(f"[FakeLens] Launching web server on http://localhost:{args.port} ...")
        run_server(port=args.port)

    elif args.command == "evaluate":
        from evaluation.evaluate_models import evaluate_model
        evaluate_model(args.dataset, args.task, args.model, split=args.split, allow_test=args.allow_test_split)

    elif args.command == "train":
        if args.model == "cnn":
            from training.cnn import train_cnn
            train_cnn(args.dataset, args.task, epochs=args.epochs, dry_run=args.dry_run)
        elif args.model == "efficientnet_b0":
            from training.efficientnet_b0 import train_efficientnet_b0
            train_efficientnet_b0(args.dataset, args.task, epochs=args.epochs, dry_run=args.dry_run)
        elif args.model == "random_forest":
            from training.random_forest import train_random_forest
            train_random_forest(args.dataset, args.task, dry_run=args.dry_run)
        elif args.model == "mlp":
            from training.mlp import train_mlp
            train_mlp(args.dataset, args.task, dry_run=args.dry_run)
        elif args.model == "knn":
            from training.knn import train_knn
            train_knn(args.dataset, args.task, dry_run=args.dry_run)

    elif args.command == "preprocess":
        from preprocessing.tabular_preprocessing import extract_features_from_image
        print(f"[FakeLens] Extracting features from: {args.image}")
        feats = extract_features_from_image(args.image)
        print(f"[FakeLens] Successfully extracted {len(feats)}-dimensional feature vector.")


if __name__ == "__main__":
    main()
