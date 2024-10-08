import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class ImageClassifier:
    def __init__(self, img_width=250, img_height=250, batch_size=32):
        """
        Initialize the ImageClassifier object.

        Args:
            img_width (int): Width of the input images.
            img_height (int): Height of the input images.
            batch_size (int): Batch size for training and evaluation.
        """
        self.results_dir: str = None
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.learning_rate = 0.0001
        self.balance_type: str = "downsampling"  # "downsampling", "upsampling" or None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
        self.class_names = []

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
            img = tf.io.read_file(image_location)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_width, self.img_height])
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
        dataset = dataset.shuffle(buffer_size=len(data))
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
        data["image_location"] = data["image_location"].apply(
            lambda x: os.path.join(directory, x)
        )

        # Compute class names from the full dataset before balancing and splitting
        self.class_names = data["genre_label"].unique().tolist()
        print(f"Class names: {self.class_names}")
        label_to_index = {label: i for i, label in enumerate(self.class_names)}
        print(f"Label to index: {label_to_index}")

        # Convert genre labels to numeric indices for the entire dataset
        data["genre_label"] = data["genre_label"].map(label_to_index)

        # Balance the dataset before splitting
        if self.balance_type == "upsampling":
            # Balance the dataset (UPSAMPLING)
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

        elif self.balance_type == "downsampling":
            # Balance the dataset (DOWNSAMPLING)
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
            balanced_data = data

        # Shuffle the balanced dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # Split the data into training, validation, and test sets
        train_data, temp_data = train_test_split(
            balanced_data, test_size=(val_size + test_size), random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=test_size / (val_size + test_size), random_state=42
        )

        # Print the number of images in each set
        print(f"Found {len(train_data)} images for training.")
        print(f"Found {len(val_data)} images for validation.")
        print(f"Found {len(test_data)} images for testing.")

        # Print the distribution of classes in each set
        print(
            f"Training set class distribution:\n{train_data['genre_label'].value_counts()}"
        )
        print(
            f"Validation set class distribution:\n{val_data['genre_label'].value_counts()}"
        )
        print(
            f"Test set class distribution:\n{test_data['genre_label'].value_counts()}"
        )

        # Prepare datasets
        self.train_dataset = self._prepare_dataset(train_data)
        self.val_dataset = self._prepare_dataset(val_data)
        self.test_dataset = self._prepare_dataset(test_data)

    def build_model(self):
        """
        Build the CNN model for image classification.
        """
        base_model = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_width, self.img_height, 3),
        )

        for layer in base_model.layers:
            layer.trainable = False

        base_model.summary()
        self.model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.class_names), activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.summary()

    def train(self, epochs=50):
        """
        Train the CNN model on the training data.

        Args:
            epochs (int): Number of epochs for training.
        """
        checkpoint_path = os.path.join(self.results_dir, "best_model.keras")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, verbose=2
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1
            ),
        ]

        # Calculate steps per epoch
        steps_per_epoch = math.ceil(
            tf.data.experimental.cardinality(self.train_dataset).numpy()
            // self.batch_size
        )
        validation_steps = math.ceil(
            tf.data.experimental.cardinality(self.val_dataset).numpy()
            // self.batch_size
        )

        for images, labels in self.train_dataset.take(
            1
        ):  # Take one batch from the dataset
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

            #  visualize some images from the batch
            plt.figure(figsize=(10, 10))
            for i in range(min(9, len(images))):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy())
                plt.title(f"Label: {labels[i].numpy()}")
                plt.axis("off")
            plt.show()

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
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

    def evaluate(self):
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

    def predict(self, img_path):
        """
        Predict the class of a single image using the trained model.

        Args:
            img_path (str): Path to the image to be classified.

        Returns:
            str: Predicted class label.
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.img_width, self.img_height])
        image = tf.expand_dims(image, 0)

        prediction = self.model.predict(image)
        predicted_class = self.train_dataset.class_names[np.argmax(prediction)]
        return predicted_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Base directory containing images"
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

    # Initialize the image classifier
    classifier = ImageClassifier(
        img_width=args.img_width, img_height=args.img_height, batch_size=args.batch_size
    )
    classifier.results_dir = args.results_dir
    classifier.balance_type = args.balance_type

    # Load training and validation data
    classifier.load_data(args.csv_file, args.directory)

    # Build and train the model
    classifier.build_model()
    classifier.train(epochs=args.epochs)

    # Evaluate the model
    classifier.evaluate()


if __name__ == "__main__":
    main()
