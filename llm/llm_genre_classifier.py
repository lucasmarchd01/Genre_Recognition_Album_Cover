import base64
import json
import logging
import os
import time
import uuid
from difflib import get_close_matches
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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

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

    def enforce_daily_limit(self):
        """Enforce API rate limits by ensuring appropriate delays."""
        with self.lock:
            current_time = time.time()

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

        count = 1
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
            self.enforce_daily_limit()

            # Predict genre
            prediction = self.predict_genre(encoded_image)
            predictions.append(prediction)

            logger.info(f"\nProcessed image {count}/{len(dataset)}: {prediction}")
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
        # logger.info("Starting predictions for training set...")
        # train_predictions = self.iterate_and_predict(train_data)

        # logger.info("Starting predictions for validation set...")
        # val_predictions = self.iterate_and_predict(val_data)

        logger.info("Starting predictions for test set...")
        test_predictions = self.iterate_and_predict(test_data)

        return {
            # "train": train_predictions,
            # "validation": val_predictions,
            "test": test_predictions,
        }

    def load_data(self, full_csv, directory, val_size=0.15, test_size=0.15) -> bool:
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

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def predict_genre(self, image) -> str:
        """
        Predict the music genre based on an album cover image using OpenAI API.

        Args:
            image (str): a base64 encoded image.

        Returns:
            str: Predicted genre from the model.
        """
        rate_limit_per_minute = self.rate_limits["RPM"]
        delay = 60.0 / rate_limit_per_minute
        time.sleep(delay)

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_genre",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "genre": {"type": "string", "enum": self.class_names},
                            },
                            "required": ["genre"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Make sure you're using a model that supports structured outputs
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"This is an image of an album cover. "
                                f"The possible music genres are: {' - '.join(self.class_names)}. "
                                f"Based on the visual style and design, predict the genre. "
                                f"Your response must strictly adhere to the predefined options.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}",
                                },
                            },
                        ],
                    },
                ],
                tools=tools,
                tool_choice="required",
            )
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            predicted_genre = arguments.get("genre")
            return self.get_genre(predicted_genre)
        except Exception as e:
            logger.exception(f"Error occurred during genre prediction: {e}")
            return "error"

    def get_genre(self, genre: str):
        """
        Returns the genre if it matches one in the class_names list.
        If not, attempts fuzzy matching to find the closest match.
        """
        if genre in self.class_names:
            return genre
        else:
            # Fuzzy matching to find the closest genre
            closest_matches = get_close_matches(genre, self.class_names, n=1, cutoff=0)
            return closest_matches[0]

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

    def encode_labels(self, labels):
        """
        Encodes a list of string labels into their corresponding integer indices.

        Args:
            labels (list): List of string labels to encode.

        Returns:
            list: List of encoded integer labels.
        """
        label_to_index = {label: i for i, label in enumerate(self.class_names)}
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
            predicted_labels = self.encode_labels(predicted_labels)

            conf_matrix = confusion_matrix(true_labels, predicted_labels)
            class_report = classification_report(
                true_labels,
                predicted_labels,
                target_names=self.class_names,
                labels=range(5),
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
                f.write(f"\n{class_report}\n")
                f.write(f"Confusion Matrix:\n{conf_matrix}\n")
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
            # "train": self.train_data,
            # "validation": self.val_data,
            "test": self.test_data,
        }

        for dataset_name, dataset in datasets.items():
            true_labels = dataset["genre_label"].tolist()
            predicted_labels = predictions[dataset_name]
            self.evaluate_and_save_results(
                dataset_name, true_labels, predicted_labels, output_dir
            )
