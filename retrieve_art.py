import pandas as pd
import csv
import argparse
import os
import logging
from typing import Optional, List, Dict
import requests


class Art:
    def __init__(self):
        """
        Initialize an Art object.

        Attributes:
            mbid (str): The MusicBrainz Identifier (MBID) of the release group.
            response_url (str): The URL of the API response.
            exists (bool): Indicates whether cover art exists for the release group.
            images (list): A list of cover art images available for the release group.
            release (str): The URL of the MusicBrainz release associated with the cover art.
            image_locations (dict): A dictionary mapping MBID to the location of downloaded images.
        """
        self.mbid: str = ""
        self.response_url: str = ""
        self.exists: bool = False
        self.images: List = []
        self.release: str = ""
        self.image_locations: Dict = {}

    def download(self):
        """
        Download front cover images for the release group.

        This method downloads the smallest-sized front cover image available for each release
        group. It stores the downloaded image file and updates the image locations dictionary
        accordingly.
        """
        for image in self.images:
            if image.get("front", False):
                thumbnails = image.get("thumbnails")
                smallest_size_url = thumbnails.get("small")
                if smallest_size_url:
                    filename = f"{self.mbid}_{image['id']}.jpg"
                    try:
                        response = requests.get(smallest_size_url)
                        if response.status_code == 200:
                            filepath = f"data/images/{filename}"
                            with open(filepath, "wb") as f:
                                f.write(response.content)
                            self.store_image_location(filepath)
                            logging.info(f"Downloaded front cover image: {filename}")
                        else:
                            logging.error(
                                f"Failed to download image: {response.status_code}"
                            )
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Error downloading image: {e}")

    def get_response(self, url: str) -> None:
        """
        Send a GET request to the specified URL and process the response.

        This method sends a GET request to the specified URL and processes the JSON response
        to extract information about cover art for the release group.

        Args:
            url (str): The URL to send the GET request to.
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                self.response_url = response.url
                self.exists = True
                self.release = data.get("release")
                self.images = data.get("images", [])
            elif response.status_code == 404:
                logging.warning(
                    f"No cover art found for release group with MBID: {self.mbid}"
                )
            else:
                logging.error(
                    f"Failed to retrieve cover art for release group with MBID: {self.mbid}. Status code: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making request to {url}: {e}")

    def store_image_location(self, filename: str) -> None:
        """
        Store the location of a downloaded image.

        This method updates the image locations dictionary with the location of a downloaded
        image file.

        Args:
            filename (str): The filename of the downloaded image.
        """
        self.image_locations[self.mbid] = filename

    def get_image_locations(self):
        """
        Get the dictionary mapping MBID to image locations.

        Returns:
            dict: A dictionary mapping MBID to the location of downloaded images.
        """
        return self.image_locations


def read_tsv(filename: str) -> pd.DataFrame:
    """
    Read data from a tab-separated values (TSV) file into a DataFrame.

    Args:
        filename (str): The filename of the TSV file to read.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the TSV file.
    """
    try:
        if os.path.exists(filename):
            return pd.read_csv(filename, sep="\t")
    except OSError as e:
        logging.error(f"File {filename} not found: {e}")
    return None


def get_art_from_tsv(filename: str) -> List[Art]:
    """
    Retrieve cover art information from a TSV file.

    This function reads data from a TSV file containing release group MBIDs and retrieves
    cover art information for each release group.

    Args:
        filename (str): The filename of the TSV file containing release group MBIDs.

    Returns:
        List[Art]: A list of Art objects containing cover art information for each release group.
    """
    df = read_tsv(filename)
    release_group = df["releasegroupmbid"]
    unique_values = release_group.unique().tolist()

    art_objects = []
    try:
        for mbid in unique_values:
            logging.info(f"retrieving art for mbid: {mbid}")
            art = Art()
            art.mbid = mbid
            art.get_response(f"https://coverartarchive.org/release-group/{art.mbid}")
            art.download()
            art_objects.append(art)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Saving collected data...")
        store_to_csv(art_objects, "data/csv/output_interrupted.csv")
        logging.info("Collected data saved.")
        raise  # Re-raise KeyboardInterrupt to exit gracefully

    return art_objects


def store_to_csv(
    art_objects: List[Art], output_filename: str = "mbid_to_image_filenames.csv"
) -> None:
    """
    Store MBID and image location mappings to a CSV file.

    Args:
        art_objects (List[Art]): List of Art objects containing image locations.
        output_filename (str): Filename for the output CSV file.
    """
    with open(output_filename, mode="w", newline="") as csvfile:
        fieldnames = ["mbid", "image_location"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for art in art_objects:
            for mbid, location in art.get_image_locations().items():
                writer.writerow({"mbid": mbid, "image_location": location})


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-d", "--download", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/art_retrieval.log",
        filemode="w",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
        level=logging.DEBUG,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info("Started")

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/csv", exist_ok=True)

    art_objects = get_art_from_tsv(args.filename)
    if args.output:
        store_to_csv(art_objects, args.output)
    else:
        store_to_csv(art_objects)


if __name__ == "__main__":
    main()
