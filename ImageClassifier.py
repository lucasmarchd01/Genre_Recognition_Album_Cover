import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
from optuna.integration.tensorboard import TensorBoardCallback
from optuna.trial import TrialState
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class ImageClassifier:
    def __init__(
        self,
        img_width=250,
        img_height=250,
        batch_size=8,
        learning_rate=0.001,
        epochs=50,
        balance_type="downsampling",
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
        self.results_dir: str = None
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
        self.class_names = []

        logger.info(
            "ImageClassifier initialized with parameters:\n"
            "  Image Size: (%d, %d)\n"
            "  Batch Size: %d\n"
            "  Learning Rate: %.5f\n"
            "  Balance Type: '%s'",
            img_width,
            img_height,
            batch_size,
            learning_rate,
            balance_type,
        )

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

        # Distribution of classes in each set
        logger.info(
            f"Training set class distribution:\n{train_data['genre_label'].value_counts()}"
        )
        logger.info(
            f"Validation set class distribution:\n{val_data['genre_label'].value_counts()}"
        )
        logger.info(
            f"Test set class distribution:\n{test_data['genre_label'].value_counts()}"
        )

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
        if trial is not None:
            base_model_name = trial.suggest_categorical(
                "base_model", ["VGG16", "ResNet50"]
            )
        else:
            base_model_name = "VGG16"

        if base_model_name == "VGG16":
            base_model = tf.keras.applications.VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_width, self.img_height, 3),
            )
        elif base_model_name == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_width, self.img_height, 3),
            )

        for layer in base_model.layers:
            layer.trainable = False

        # Hyperparameter tuning for dense layers
        model = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])

        # Add dense layers
        if trial is not None:
            num_dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
            dense_units = trial.suggest_int("dense_units", 64, 512)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        else:
            num_dense_layers = 2
            dense_units = 128
            dropout_rate = 0.5
            learning_rate = self.learning_rate

        for _ in range(num_dense_layers):
            model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(len(self.class_names), activation="softmax"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        self.model.summary()

    def run_study(self, n_trials=50):
        """
        Run the Optuna study for hyperparameter tuning.

        Args:
            n_trials (int): Number of trials for the hyperparameter search.
        """
        study = optuna.create_study(direction="maximize")
        tensorboard_callback = TensorBoardCallback("logs/", metric_name="accuracy")
        study.optimize(
            self.objective, n_trials=n_trials, callbacks=[tensorboard_callback]
        )

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: ", len(study.trials))
        logger.info("  Number of pruned trials: ", len(pruned_trials))
        logger.info("  Number of complete trials: ", len(complete_trials))
        # Print the best trial
        trial = study.best_trial
        logger.info(f"Best trial: {trial.number}")
        logger.info(f"Best trial params: {trial.params}")
        logger.info(f"Best trial accuracy: {trial.value}")

        return trial

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.
        """
        self.build_model(trial)

        # Define callbacks for early stopping and learning rate reduction
        checkpoint_path = os.path.join(self.results_dir, "best_model.keras")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, verbose=2
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=7, min_lr=1e-6, verbose=1
            ),
        ]

        # Fit the model
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0,
        )
        # Evaluate the model on the validation set
        val_loss, val_accuracy = self.model.evaluate(self.val_dataset, verbose=0)
        return val_accuracy  # Optuna maximizes accuracy

    def save_model(self, path) -> None:
        self.model.save(path)

    def load_model(self, path) -> None:
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
                    monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1
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
            verbose=10,
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
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        class_labels = self.class_names

        y_pred = np.argmax(self.model.predict(self.test_dataset), axis=1)
        y_true = np.concatenate([y for _, y in self.test_dataset], axis=0)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.savefig(os.path.join(self.results_dir, "confusion_matrix.png"))
        plt.close()

        report = classification_report(y_true, y_pred, target_names=class_labels)

        with open(os.path.join(self.results_dir, "results.txt"), "w") as f:
            f.write(
                f"Validation Loss: {test_loss}, Validation Accuracy: {test_accuracy}\n"
            )
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
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.img_width, self.img_height])
        image = tf.expand_dims(image, 0)

        prediction = self.model.predict(image)
        predicted_class = self.train_dataset.class_names[np.argmax(prediction)]
        return predicted_class
