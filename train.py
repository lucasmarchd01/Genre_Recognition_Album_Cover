import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class ImageClassifier:
    def __init__(self, img_width=250, img_height=250, batch_size=32):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.model = None

    def load_data(self, directory):
        csv_filename = os.path.join(directory, "data", "csv", "final_top_6.csv")
        data = pd.read_csv(csv_filename)

        # Adjust image locations in the CSV file
        data["image_location"] = data["image_location"].apply(
            lambda x: os.path.join(directory, x)
        )
        print(data.head())

        # Balance the dataset
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

        # Split the balanced dataset into train, validation, and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42
        )

        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            directory=None,
            x_col="image_location",
            y_col="genre_label",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_data,
            directory=None,
            x_col="image_location",
            y_col="genre_label",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_data,
            directory=None,
            x_col="image_location",
            y_col="genre_label",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

    def build_model(self):
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_width, self.img_height, 3),
        )

        for layer in base_model.layers:
            layer.trainable = False

        self.model = Sequential(
            [
                base_model,
                Flatten(),
                Dense(256, activation="relu"),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(len(self.train_generator.class_indices), activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, epochs=2):
        checkpoint_path = "results/best_model.keras"
        checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True)

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.n // self.batch_size,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.n // self.batch_size,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                checkpoint_callback,
            ],
        )

        # Save training and validation accuracy plots
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("results/training_validation_accuracy.png")
        plt.close()

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)

        # Get class labels
        class_labels = list(self.train_generator.class_indices.keys())

        # Predict test data
        y_pred = np.argmax(self.model.predict(self.test_generator), axis=1)
        y_true = self.test_generator.classes

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
        plt.savefig("results/confusion_matrix.png")
        plt.close()

        # Compute and print classification report
        report = classification_report(y_true, y_pred, target_names=class_labels)

        # Save results to a text file
        with open("results/results.txt", "w") as f:
            f.write(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")


def main(args):
    os.makedirs("results", exist_ok=True)
    classifier = ImageClassifier()
    classifier.load_data(args.directory)
    classifier.build_model()
    classifier.train()
    classifier.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to the directory")
    args = parser.parse_args()
    main(args)
