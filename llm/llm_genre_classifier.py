import base64
import json
import logging
import os
import time
import uuid
from queue import Queue
from threading import Lock
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logger = logging.getLogger(__name__)
client = OpenAI()


class LLMClassifier:
    def __init__(self, id=None) -> None:
        self.id = id if id else uuid.uuid4()
        self.dataframe = pd.DataFrame()
        self.class_names = [
            "electronic",
            "rock",
            "folk, world, & country",
            "pop",
            "jazz",
        ]
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.balance_type = "downsampling"
        self.rate_limits = {
            "RPM": 500,  # Requests per minute
            "RPD": 10000,  # Requests per day
            "TPM": 200000,  # Tokens per minute
        }
        self.last_request_time = 0
        self.lock = Lock()
        self.request_queue = Queue()

    def enforce_rate_limits(self):
        """Enforce API rate limits by ensuring appropriate delays."""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time

            # Calculate wait time based on requests per minute
            min_interval = 60 / self.rate_limits["RPM"]
            if elapsed < min_interval:
                logger.info(
                    f"Sleeping to respect RPM limit... {min_interval - elapsed:.2f} seconds"
                )
                time.sleep(min_interval - elapsed)

            # Track daily request limit
            if self.request_queue.qsize() >= self.rate_limits["RPD"]:
                logger.error("Daily request limit reached. Pausing until the next day.")
                time_to_next_day = 86400 - (current_time % 86400)
                time.sleep(time_to_next_day)

            # Record the time of this request
            self.last_request_time = time.time()
            self.request_queue.put(current_time)

    def iterate_and_predict(self, dataset):
        """
        Iterate through a dataset and predict genres using the OpenAI API.

        Args:
            dataset (pd.DataFrame): A DataFrame containing image paths and genre labels.

        Returns:
            list: A list of predicted genres.
        """
        predictions = []

        count = 0
        for idx, row in dataset.iterrows():
            image_path = row["image_location"]

            # Encode the image
            try:
                encoded_image = self.encode_image(image_path)
            except FileNotFoundError:
                logger.error(f"File not found: {image_path}")
                predictions.append("File not found")
                continue

            # Enforce rate limits
            self.enforce_rate_limits()

            # Predict genre
            prediction = self.predict_genre(encoded_image)
            predictions.append(prediction)
            print(predictions)

            logger.info(f"Processed image {count}/{len(dataset)}: {prediction}")
            count += 1

        return predictions

    def process_datasets(self, train_data, val_data, test_data):
        """
        Process all datasets: training, validation, and testing.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            test_data (pd.DataFrame): Test dataset.

        Returns:
            dict: Predicted genres for each dataset.
        """
        logger.info("Starting predictions for training set...")
        train_predictions = self.iterate_and_predict(train_data)

        logger.info("Starting predictions for validation set...")
        val_predictions = self.iterate_and_predict(val_data)

        logger.info("Starting predictions for test set...")
        test_predictions = self.iterate_and_predict(test_data)

        return {
            "train": train_predictions,
            "validation": val_predictions,
            "test": test_predictions,
        }

    def load_data(self, full_csv, directory, val_size=0.15, test_size=0.15):
        """
        Load training, validation, and test data from a full CSV file.

        Args:
            full_csv (str): Path to the full CSV file.
            directory (str): Base directory for image files.
            val_size (float): Proportion of data to use for validation.
            test_size (float): Proportion of data to use for testing.
        """
        try:
            data = pd.read_csv(full_csv)

            # Adjust image locations in the CSV file
            data["image_location"] = data["image_location"].apply(
                lambda x: os.path.join(directory, x)
            )

            # Compute class names from the full dataset before balancing and splitting
            self.class_names = data["genre_label"].unique().tolist()
            logger.info(f"Class names: {self.class_names}")
            label_to_index = {label: i for i, label in enumerate(self.class_names)}
            logger.info(f"Label to index: {label_to_index}")

            # Convert genre labels to numeric indices for the entire dataset
            data["genre_label"] = data["genre_label"].map(label_to_index)

            self.dataframe = data

            # Balance the dataset
            balanced_data = self.balance_dataset(self.balance_type)

            # Shuffle the dataset
            balanced_data_shuffled = balanced_data.sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            self.dataframe = balanced_data_shuffled

            # Split the data into training, validation, and test sets
            # We are splitting the dataset similarly to the ImageClassifier class so that we can compare
            # results on predicting the same images.
            train_data, temp_data = train_test_split(
                balanced_data, test_size=(val_size + test_size), random_state=42
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=test_size / (val_size + test_size), random_state=42
            )

            # Number of images in each set
            logger.info(f"Found {len(train_data)} images for training.")
            self.train_data = train_data

            logger.info(f"Found {len(val_data)} images for validation.")
            self.val_data = val_data

            logger.info(f"Found {len(test_data)} images for testing.")
            self.test_data = test_data
            return True
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            return False

    def balance_dataset(self, balance_type="none") -> Optional[pd.DataFrame]:
        """
        Balance the dataset based on the balance type (upsampling or downsampling).

        Args:
            balance_type (str): The type of balancing to perform ("upsampling", "downsampling", or "none").

        Returns:
            DataFrame: The balanced dataset.
        """
        data = self.dataframe

        if balance_type == "upsampling":
            balanced_data = pd.DataFrame()
            for label in data["genre_label"].unique():
                label_data = data[data["genre_label"] == label]
                balanced_data = pd.concat(
                    [
                        balanced_data,
                        resample(
                            label_data,
                            replace=True,
                            n_samples=len(data) // data["genre_label"].nunique(),
                            random_state=42,
                        ),
                    ]
                )

        elif balance_type == "downsampling":
            balanced_data = pd.DataFrame()
            minority_size = data["genre_label"].value_counts().min()
            for label in data["genre_label"].unique():
                label_data = data[data["genre_label"] == label]
                if len(label_data) > minority_size:
                    label_data = resample(
                        label_data,
                        replace=False,
                        n_samples=minority_size,
                        random_state=42,
                    )
                balanced_data = pd.concat([balanced_data, label_data])

        else:
            logger.info("No balancing applied.")
            balanced_data = data

        return balanced_data

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def predict_genre(self, image):
        """
        Predict the music genre based on an album cover image using OpenAI API.

        Args:
            image (str): a base 64 encoded image

        Returns:
            str: Predicted genre from the model.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"This is an image of an album cover. "
                                    f"The possible music genres are: {self.class_names}. "
                                    f"Based on the visual style and design, predict the genre. "
                                    f"Only predict the genre with no explanation."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error occurred: {e}"

    def save_results_to_file(self, results, file_path):
        """
        Save the prediction results to a JSON file.

        Args:
            results (dict): The prediction results.
            file_path (str): Path to the file where results should be saved.
        """
        try:
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def encode_labels(labels, class_names):
        """
        Encodes a list of string labels into their corresponding integer indices.

        Args:
            labels (list): List of string labels to encode.
            class_names (list): List of all class names in the dataset.

        Returns:
            list: List of encoded integer labels.
        """
        label_to_index = {label: i for i, label in enumerate(class_names)}
        try:
            encoded_labels = [label_to_index[label] for label in labels]
            return encoded_labels
        except KeyError as e:
            logger.error(f"Label {e} not found in class names.")
            raise ValueError(f"Invalid label: {e}")

    def evaluate_and_save_results(
        self, dataset_name, true_labels, predicted_labels, output_dir
    ):
        """
        Evaluate prediction results and save the confusion matrix and classification report.

        Args:
            dataset_name (str): The name of the dataset being evaluated (e.g., "train", "val", "test").
            true_labels (list): The ground truth labels.
            predicted_labels (list): The predicted labels.
            output_dir (str): Directory where the output files will be saved.
        """
        try:
            # Generate confusion matrix and classification report
            predicted_labels = self.encode_labels(predicted_labels, self.class_names)

            conf_matrix = confusion_matrix(true_labels, predicted_labels)
            class_report = classification_report(
                true_labels, predicted_labels, target_names=self.class_names
            )

            # Save confusion matrix as an image
            plt.figure(figsize=(10, 8))
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="g")
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.title("Confusion Matrix")
            plt.xticks(
                ticks=np.arange(len(self.class_names)) + 0.5, labels=self.class_names
            )
            plt.yticks(
                ticks=np.arange(len(self.class_names)) + 0.5, labels=self.class_names
            )
            conf_matrix_path = f"{output_dir}/{dataset_name}_confusion_matrix.png"
            plt.savefig(conf_matrix_path)
            plt.close()
            logger.info(f"Confusion matrix saved to {conf_matrix_path}")

            # Save classification report as a text file
            report_path = f"{output_dir}/{dataset_name}_classification_report.txt"
            with open(report_path, "w") as f:
                f.write(
                    f"Classification Report for {dataset_name.capitalize()} Dataset\n"
                )
                f.write(class_report)
                f.write("Confusion matrix")
                f.write(conf_matrix)
            logger.info(f"Classification report saved to {report_path}")

        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name} dataset: {e}")

    def evaluate_all_datasets(self, predictions, output_dir):
        """
        Evaluate and save results for training, validation, and test datasets.

        Args:
            predictions (dict): Dictionary containing predicted genres for each dataset.
            output_dir (str): Directory where evaluation outputs will be saved.
        """
        datasets = {
            "train": self.train_data,
            "validation": self.val_data,
            "test": self.test_data,
        }

        for dataset_name, dataset in datasets.items():
            true_labels = dataset["genre_label"].tolist()
            predicted_labels = predictions[dataset_name]
            self.evaluate_and_save_results(
                dataset_name, true_labels, predicted_labels, output_dir
            )
