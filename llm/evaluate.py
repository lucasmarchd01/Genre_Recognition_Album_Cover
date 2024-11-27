from llm_genre_classifier import LLMClassifier


if __name__ == "__main__":

    # Path to csv
    csv_path = "data/csv/csv_discogs/final_top_5_discogs.csv"

    llm_classifier = LLMClassifier()

    llm_classifier.load_data(csv_path, "./")

    predictions = llm_classifier.process_datasets(
        llm_classifier.train_data,
        llm_classifier.val_data,
        llm_classifier.test_data,
    )

    evaluation = llm_classifier.evaluate_all_datasets(
        predictions, f"results_{llm_classifier.id}"
    )
