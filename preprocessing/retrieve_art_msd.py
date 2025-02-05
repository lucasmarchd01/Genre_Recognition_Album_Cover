import os
import requests
import pandas as pd
from urllib.parse import urlparse


def download_image(image_url, save_folder):
    """Downloads an image from the given URL and saves it to the specified folder."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            # Extract filename from URL
            filename = os.path.basename(urlparse(image_url).path)
            save_path = os.path.join(save_folder, filename)

            with open(save_path, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            return filename
        else:
            print(f"Failed to download {image_url}: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
    return None


def process_tsv(input_tsv, output_tsv, image_folder):
    """Reads a TSV file, extracts unique image URLs, downloads images, and saves an updated TSV."""
    os.makedirs(image_folder, exist_ok=True)

    df = pd.read_csv(input_tsv, sep="\t")

    # Extract unique image URLs
    unique_images = df[["image_url"]].drop_duplicates().reset_index(drop=True)

    # Download images
    unique_images["image_filename"] = unique_images["image_url"].apply(
        lambda url: download_image(url, image_folder)
    )

    # Remove entries where the download failed
    unique_images = unique_images.dropna(subset=["image_filename"])

    # Save updated TSV
    unique_images.to_csv(output_tsv, sep="\t", index=False)

    print(f"Processed TSV saved to: {output_tsv}")
    print(f"Images downloaded to: {image_folder}")


def map_genre_set(original_tsv, unique_tsv, mapped_tsv):
    """Maps genre and set from the original TSV to the unique image dataset."""
    original_df = pd.read_csv(original_tsv, sep="\t")
    unique_df = pd.read_csv(unique_tsv, sep="\t")

    # Merge to retain genre and set information
    mapped_df = unique_df.merge(
        original_df[["image_url", "genre", "set"]].drop_duplicates(),
        on="image_url",
        how="left",
    )

    # Save updated TSV
    mapped_df.to_csv(mapped_tsv, sep="\t", index=False)
    print(f"Mapped TSV saved to: {mapped_tsv}")


def update_image_locations(input_csv, output_csv=None):
    """
    Reads a CSV file, appends 'data/images_msd/' to each value in the 'image_location' column,
    and writes the updated data to a new CSV file (or overwrites the original if no output_csv is provided).

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str, optional): Path to save the updated CSV file. If not provided, the input file will be overwritten.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Check if 'image_location' column exists
    if "image_location" not in df.columns:
        raise ValueError("The CSV file does not contain an 'image_location' column.")

    # Append the prefix to each image_location entry. Here, we're prepending the string.
    # Adjust the logic if you meant to append it to the end instead.
    df["image_location"] = "data/images_msd/" + df["image_location"].astype(str)

    # Determine the output CSV path
    output_csv = output_csv or input_csv

    # Save the updated DataFrame back to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download unique album cover images and update TSV file."
    )
    parser.add_argument("input_tsv", help="Path to the input TSV file.")
    parser.add_argument("output_tsv", help="Path to save the updated TSV file.")
    # parser.add_argument("image_folder", help="Folder to save downloaded images.")

    args = parser.parse_args()

    # process_tsv(args.input_tsv, args.output_tsv, args.image_folder)

    # map_genre_set(args.input_tsv, args.output_tsv, args.image_folder)
    update_image_locations(args.input_tsv, args.output_tsv)
