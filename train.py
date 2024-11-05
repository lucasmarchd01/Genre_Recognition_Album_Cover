import argparse
import logging
import os
import uuid

from ImageClassifier import ImageClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "--directory", type=str, default="./", help="Base directory containing data"
    )
    parser.add_argument(
        "--img_width", type=int, default=250, help="Width of input images"
    )
    parser.add_argument(
        "--img_height", type=int, default=250, help="Height of input images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training"
    )
    parser.add_argument(
        "--balance_type",
        type=str,
        choices=["downsampling", "upsampling", "none"],
        default="downsampling",
        help="Balance type for training data",
    )

    args = parser.parse_args()
    ID = uuid.uuid4()

    # Create results directory if it does not exist
    results_dir = f"results_{ID}"
    os.makedirs(results_dir, exist_ok=True)

    # Set up the logger
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"classifier_{ID}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting the training process...")

    # Initialize the image classifier
    classifier = ImageClassifier(
        id=ID,
        img_width=args.img_width,
        img_height=args.img_height,
        batch_size=args.batch_size,
        epochs=args.epochs,
        balance_type=args.balance_type,
    )
    classifier.results_dir = results_dir
    # Load training and validation data
    classifier.load_data(args.csv_file, args.directory)

    # Build and train a single model (uncomment)
    # classifier.build_model()
    # classifier.train(use_early_stopping=False, use_reduce_lr=False)

    # Run optuna study
    classifier.run_study(n_trials=15)

    # Evaluate the model
    classifier.evaluate()


if __name__ == "__main__":
    main()
