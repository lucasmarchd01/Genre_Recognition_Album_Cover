import logging
import os
import uuid
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
from optuna.integration.tensorboard import TensorBoardCallback
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class ImageClassifier:
    def __init__(
        self,
        id=None,
        img_width=250,
        img_height=250,
        batch_size=8,
        learning_rate=0.0001,
        epochs=50,
        balance_type="",
    ):
        """
        Initialize the ImageClassifier object.

        Args:
            img_width (int): Width of the input images.
            img_height (int): Height of the input images.
            batch_size (int): Batch size for training and evaluation.
            learning_rate (float): Learning rate for training.
            epochs (int): Number of epochs for training
            balance_type (str): Method for balancing training data.
        """
        self.id = id if id else uuid.uuid4()
        self.results_dir: str = f"results_{self.id}"
        self.img_width: int = img_width
        self.img_height: int = img_height
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.balance_type: str = balance_type  # "downsampling", "upsampling" or None
        self.dataframe: Optional[pd.DataFrame] = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
        # These are the class names used for the discogs dataset
        self.class_names = [
            "electronic",
            "rock",
            "folk, world, & country",
            "pop",
            "jazz",
        ]

        logger.info(f"Image classifier initialized with ID: {self.id}")

    def _prepare_dataset(self, data):
        """
        Prepare a TensorFlow dataset from DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing image paths and labels.

        Returns:
            tf.data.Dataset: Prepared dataset.
        """
        # Keep only the image location and genre label columns
        data = data[["image_location", "genre_label"]]

        def load_image_and_label(image_location, label):
            try:
                img = tf.io.read_file(image_location)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.image.resize(img, [self.img_width, self.img_height])
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"Error loading image {image_location}: {e}")
                img = tf.zeros((self.img_width, self.img_height, 3))
            return img, label

        # Create TensorFlow dataset from image paths and labels
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                data["image_location"].values,
                data["genre_label"].values,
            )
        )

        # Apply image processing function
        dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle, batch, and prefetch the dataset
        dataset = dataset.shuffle(seed=42, buffer_size=len(data))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def load_data(self, full_csv, directory, val_size=0.15, test_size=0.15):
        """
        Load training, validation, and test data from a full CSV file.

        Args:
            full_csv (str): Path to the full CSV file.
            directory (str): Base directory for image files.
            val_size (float): Proportion of data to use for validation.
            test_size (float): Proportion of data to use for testing.
        """
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
        train_data, temp_data = train_test_split(
            balanced_data, test_size=(val_size + test_size), random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=test_size / (val_size + test_size), random_state=42
        )

        # Number of images in each set
        logger.info(f"Found {len(train_data)} images for training.")
        logger.info(f"Found {len(val_data)} images for validation.")
        logger.info(f"Found {len(test_data)} images for testing.")

        # Prepare datasets
        self.train_dataset = self._prepare_dataset(train_data)
        self.val_dataset = self._prepare_dataset(val_data)
        self.test_dataset = self._prepare_dataset(test_data)

    def balance_dataset(self, balance_type="none") -> Optional[pd.DataFrame]:
        """
        Balance the dataset based on the balance type (upsampling or downsampling).

        Args:
            balance_type (str): The type of balancing to perform ("upsampling", "downsampling", or "none").

        Returns:
            DataFrame: The balanced dataset.
        """

        # Check if the dataframe exists
        if not hasattr(self, "dataframe"):
            logger.error(
                "No data has been loaded. Please call `load_data` before balancing the dataset."
            )
            return None

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

    def build_model(self, trial=None):
        """
        Build the CNN model for image classification with optuna integration for hyperparameters.
        """
        ###
        # Uncomment below if you wish to use a dynamic approach to selecting the base model
        # and hyperparameters using Optuna.
        ###
        # # Define a dictionary for available base models
        # base_model_dict = {
        #     "VGG16": tf.keras.applications.VGG16,
        #     "ResNet50": tf.keras.applications.ResNet50,
        #     "InceptionV3": tf.keras.applications.InceptionV3,
        #     "Xception": tf.keras.applications.Xception,
        #     "MobileNetV2": tf.keras.applications.MobileNetV2,
        # }

        # # Suggest base model using Optuna or default to VGG16
        # if trial is not None:
        #     base_model_name = trial.suggest_categorical(
        #         "base_model", list(base_model_dict.keys())
        #     )
        # else:
        #     base_model_name = "VGG16"

        # # Dynamically load the selected base model
        # base_model_class = base_model_dict[base_model_name]
        base_model = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_width, self.img_height, 3),
        )

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Build the model
        model = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])

        # Add dense layers with Optuna suggestions or default values
        if trial is not None:
            num_dense_layers = trial.suggest_int("num_dense_layers", 1, 4)
            dense_units = trial.suggest_int("dense_units", 64, 1024)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.8)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        else:
            num_dense_layers = 2
            dense_units = 128
            dropout_rate = 0.5
            learning_rate = self.learning_rate

        # Add dense layers and dropout
        for _ in range(num_dense_layers):
            model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        # Output layer
        model.add(tf.keras.layers.Dense(len(self.class_names), activation="softmax"))

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        self.model.summary()

    def run_study(self, n_trials=15):
        """
        Run the Optuna study for hyperparameter tuning.

        Args:
            n_trials (int): Number of trials for the hyperparameter search.

        """
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction="maximize")
        tensorboard_callback = TensorBoardCallback(
            f"logs/fit/{self.id}", metric_name="accuracy"
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            gc_after_trial=True,
            callbacks=[tensorboard_callback],
        )

        # Print the best trial
        trial = study.best_trial
        logger.info(f"Best trial: {trial.number}")
        logger.info(f"Best trial params: {trial.params}")
        logger.info(f"Best trial accuracy: {trial.value}")

        self.load_model(
            os.path.join(self.results_dir, f"trial-{trial.number}_best_model.keras")
        )
        return trial

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.
        """
        self.build_model(trial)

        # Define callbacks for early stopping and learning rate reduction
        checkpoint_path = os.path.join(
            self.results_dir, f"trial-{trial.number}_best_model.keras"
        )
        log_dir = f"logs/fit/{self.id}/trial-{trial.number}"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, verbose=2
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=7, min_lr=1e-6, verbose=2
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        ]

        # Fit the model
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=2,
        )
        # Evaluate the model on the validation set
        val_loss, val_accuracy = self.model.evaluate(self.val_dataset, verbose=2)
        return val_accuracy  # Optuna maximizes accuracy

    def save_model(self, path) -> None:
        """Save a keras model.

        Args:
            path (str): Path to save model.
        """
        self.model.save(path)

    def load_model(self, path) -> None:
        """Load a keras model.

        Args:
            path (str): Path to keras model.
        """
        self.model = tf.keras.models.load_model(path)

    def train(
        self, use_early_stopping=True, use_reduce_lr=True, use_tensorboard=True
    ) -> None:
        """
        Train the CNN model on the training data.

        Args:
            epochs (int): Number of epochs for training.
        """
        checkpoint_path = os.path.join(self.results_dir, "best_model.keras")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        ]
        if use_early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, verbose=2
                )
            )

        if use_reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=2
                )
            )
        if use_tensorboard:
            log_dir = "logs/fit/"
            callbacks.append(
                tf.keras.callbacks.Tensorboard(log_dir=log_dir, histogram_freq=1)
            )

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        # Plot training history
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "training_validation_accuracy.png"))
        plt.close()

        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "training_validation_loss.png"))
        plt.close()

    def evaluate(self) -> None:
        """
        Evaluate the trained model on the validation data and generate evaluation metrics.
        """
        logger.info("Starting model evaluation...")
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        class_labels = self.class_names

        y_pred = []  # store predicted labels
        y_true = []  # store true labels

        # Since we are shuffling, iterate over the dataset
        for (
            image_batch,
            label_batch,
        ) in self.test_dataset:
            y_true.append(label_batch)
            preds = self.model.predict(image_batch)
            y_pred.append(np.argmax(preds, axis=-1))

        # convert the true and predicted labels into tensors
        y_true = tf.concat([item for item in y_true], axis=0)
        y_pred = tf.concat([item for item in y_pred], axis=0)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="g")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.savefig(os.path.join(self.results_dir, "confusion_matrix.png"))
        plt.close()

        report = classification_report(y_true, y_pred, target_names=class_labels)

        with open(os.path.join(self.results_dir, "results.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

    def predict(self, img_path) -> str:
        """
        Predict the class of a single image using the trained model.

        Args:
            img_path (str): Path to the image to be classified.

        Returns:
            str: Predicted class label.
        """
        try:
            print(f"Image path: {img_path}")
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_width, self.img_height])
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Error loading image {img_path}: {e}")
            img = tf.zeros((self.img_width, self.img_height, 3))
        image = tf.expand_dims(img, 0)

        prediction = self.model.predict(image)
        predicted_class = self.class_names[np.argmax(prediction)]
        return predicted_class
