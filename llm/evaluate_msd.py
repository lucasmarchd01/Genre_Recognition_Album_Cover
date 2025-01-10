import logging
import os
import pandas as pd
import uuid

from llm_genre_classifier import LLMClassifier


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Create results directory if it does not exist
    # ID = uuid.uuid4()
    ID = "3123bb4d-05bf-41fa-9a1f-e8183b4ed67a"
    results_dir = f"results/llm/results_{ID}"
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
    csv_path = "/Users/lucasmarch/Projects/Genre_Recognition_Album_Cover/data/csv/csv_msd/csv_msd.csv"
    dataset = pd.read_csv(csv_path)

    llm_classifier = LLMClassifier(id=ID)
    llm_classifier.class_names = dataset["genre_label"].unique().tolist()
    llm_classifier.train_data = dataset[dataset["set"] == "train"]
    llm_classifier.val_data = dataset[dataset["set"] == "val"]
    llm_classifier.test_data = dataset[dataset["set"] == "test"]

    # predictions = llm_classifier.process_datasets(
    #     llm_classifier.train_data,
    #     llm_classifier.val_data,
    #     llm_classifier.test_data,
    # )

    # llm_classifier.save_results_to_file(
    #     predictions, f"{results_dir}/results_{llm_classifier.id}.json"
    # )

    predictions = llm_classifier.load_results_from_file(
        "/Users/lucasmarch/Projects/Genre_Recognition_Album_Cover/results/llm/results_3123bb4d-05bf-41fa-9a1f-e8183b4ed67a/results_3123bb4d-05bf-41fa-9a1f-e8183b4ed67a.json"
    )

    evaluation = llm_classifier.evaluate_all_datasets(predictions, results_dir)
