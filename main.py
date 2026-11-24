# FakeLens — Multi-Domain Image Forgery Detection & Attribution Platform
# Main execution for the CLI entry point for training, evaluation, preprocessing, and server execution

import os
import sys
import argparse


# to parse command-line arguments and route subcommands
def main():
    parser = argparse.ArgumentParser(
        description="FakeLens: Multi-Domain AI Image Forgery Detection & Attribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch the web application server
  python main.py serve --port 8000

  # Evaluate a frozen model on the validation split
  python main.py evaluate --dataset deepfakeface --task binary --model efficientnet_b0 --split val
  python main.py evaluate --dataset artifact --task multiclass --model cnn --split val

  # Inspect or dry-run training configuration
  python main.py train --dataset deepfakeface --task binary --model cnn --dry_run
  python main.py train --dataset artifact --task multiclass --model efficientnet_b0 --dry_run

  # Extract 1,798-dim handcrafted features from an image
  python main.py preprocess --image path/to/image.jpg
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Server Subcommand
    serve_parser = subparsers.add_parser("serve", help="Run the FakeLens web application server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")

    # 2. Evaluate Subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    eval_parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact", "cifake"], required=True)
    eval_parser.add_argument("--task", type=str, choices=["binary", "multiclass"], required=True)
    eval_parser.add_argument("--model", type=str, required=True, help="Model ID (e.g. efficientnet_b0, cnn, random_forest, mlp, knn)")
    eval_parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    eval_parser.add_argument("--allow_test_split", action="store_true", help="Explicit authorization for protected test set")

    # 3. Train Subcommand
    train_parser = subparsers.add_parser("train", help="Inspect or train a model family")
    train_parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact", "cifake"], required=True)
    train_parser.add_argument("--task", type=str, choices=["binary", "multiclass"], required=True)
    train_parser.add_argument("--model", type=str, choices=["efficientnet_b0", "cnn", "random_forest", "mlp", "knn"], required=True)
    train_parser.add_argument("--dry_run", action="store_true", help="Validate setup without gradient updates")
    train_parser.add_argument("--epochs", type=int, default=20)

    # 4. Preprocess Subcommand
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
