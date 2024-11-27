import logging
import os
import uuid

from llm_genre_classifier import LLMClassifier

if __name__ == "__main__":

    # Create results directory if it does not exist
    ID = uuid.uuid4()
    results_dir = f"results_{ID}"
    os.makedirs(results_dir, exist_ok=True)

    # Set up the logger
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"llm_classifier_{ID}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting the evaluation process...")

    # Path to csv
    csv_path = "data/csv/csv_discogs/final_top_5_discogs.csv"

    llm_classifier = LLMClassifier(id=ID)

    llm_classifier.load_data(csv_path, "./")

    predictions = llm_classifier.process_datasets(
        llm_classifier.train_data,
        llm_classifier.val_data,
        llm_classifier.test_data,
    )

    evaluation = llm_classifier.evaluate_all_datasets(
        predictions, f"results_{llm_classifier.id}"
    )
