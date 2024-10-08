import argparse
import logging
import os

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
        "--results_dir", type=str, default="results", help="Directory to save results"
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
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
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

    # Create results directory if it does not exist
    count = 1
    while os.path.exists(args.results_dir):
        args.results_dir = f"results_{count}"
        count += 1
    os.makedirs(args.results_dir, exist_ok=True)

    # Set up the logger
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", "classifier.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting the training process...")

    # Initialize the image classifier
    classifier = ImageClassifier(
        img_width=args.img_width,
        img_height=args.img_height,
        batch_size=args.batch_size,
        epochs=args.epochs,
        balance_type=args.balance_type,
    )
    classifier.results_dir = args.results_dir
    # Load training and validation data
    classifier.load_data(args.csv_file, args.directory)

    # Build and train the model
    classifier.build_model()
    # classifier.train(use_early_stopping=False, use_reduce_lr=False)
    classifier.run_study()

    # Evaluate the model
    classifier.evaluate()


if __name__ == "__main__":
    main()
