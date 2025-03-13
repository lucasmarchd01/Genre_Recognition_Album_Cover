import musicbrainzngs
import requests
import json
import pandas as pd
import discogs_client
from discogs_client.exceptions import HTTPError
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Album:
    def __init__(self, mbid, title, artist, release_date, primary_type, genre_mb=""):
        self.mbid = mbid
        self.title = title
        self.artist = artist
        self.release_date = release_date
        self.primary_type = primary_type
        self.genre_mb = genre_mb
        self.genre_discogs = ""
        self.cover_url = f"https://coverartarchive.org/release-group/{mbid}"

    def to_dict(self):
        return {
            "mbid": self.mbid,
            "title": self.title,
            "artist": self.artist,
            "release_date": self.release_date,
            "primary_type": self.primary_type,
            "tags_musicbrainz": self.genre_mb,
            "genres_discogs": self.genre_discogs,
            "cover_url": self.cover_url,
        }


class MusicBrainzClient:
    """
    A class to interact with the MusicBrainz API using the musicbrainzngs Python package.
    """

    def __init__(self):
        # Set a user agent as required by MusicBrainz API
        musicbrainzngs.set_useragent(
            "genre-recognition", "1.0", "lucas.march@mail.mcgill.ca"
        )

    def search_albums(
        self,
        start_date="2024-07-19",
        end_date=datetime.today().strftime("%Y-%m-%d"),
        limit=25,
    ):
        """
        Retrieve albums released from start_date onwards using the MusicBrainz API.
        Uses pagination to fetch all results.

        :param start_date: Start of the release date range (YYYY-MM-DD)
        :param end_date: End of the release date range (YYYY-MM-DD)
        :param limit: Number of results per request (max 100)
        :return: List of albums with MBID, title, artist, release date, and genres
        """
        offset = 0
        albums = []

        while True:
            try:
                result = musicbrainzngs.search_release_groups(
                    query=f"firstreleasedate:[{start_date} TO {end_date}] AND primarytype:Album",
                    limit=limit,
                    offset=offset,
                )

                release_groups = result.get("release-group-list", [])

                if not release_groups:  # Stop if no more results
                    break

                for release_group in release_groups:
                    mbid = release_group.get("id")
                    title = release_group.get("title")
                    release_date = release_group.get("first-release-date", "Unknown")
                    primary_type = release_group.get("primary-type", "Album")

                    # Extract artist safely
                    artist = "Unknown"
                    if (
                        "artist-credit" in release_group
                        and release_group["artist-credit"]
                    ):
                        artist = release_group["artist-credit"][0]["artist"]["name"]

                    # Extract genres from tags
                    genres = []
                    tags = musicbrainzngs.get_release_group_by_id(
                        mbid, includes=["tags"]
                    )
                    if "tag-list" in tags["release-group"]:
                        genres = [
                            tag["name"] for tag in tags["release-group"]["tag-list"]
                        ]

                    albums.append(
                        Album(mbid, title, artist, release_date, primary_type, genres)
                    )

                offset += limit  # Move to the next page
            except musicbrainzngs.WebServiceError as e:
                logging.error(f"MusicBrainz API error: {e}")

        return albums


class DiscogsClient:
    def __init__(self, user_token):
        self.client = discogs_client.Client(
            "genre-recognition/1.0", user_token=user_token
        )

    def get_genre(self, album: Album):
        """Search Discogs for an album and get its genre."""
        try:
            results = self.client.search(
                album.title, type="release", artist=album.artist
            )
            if results and results[0].genres:
                logging.info(f"getting genres for result {results[0]}")
                album.genre_discogs = results[0].genres
            else:
                album.genre_discogs = ""
        except HTTPError as e:
            logging.error(f"HTTP error occurred: {e}")
            album.genre_discogs = ""
        except Exception as e:
            logging.error(f"Unexpected error fetching genre from Discogs: {e}")
            album.genre_discogs = ""


class CoverArtClient:
    @staticmethod
    def fetch_cover_art(mbid):
        """
        Retrieve the front cover image URL from the Cover Art Archive API for a given release group.

        :param mbid: The MusicBrainz ID of the release group.
        :return: URL of the front cover image if available, else None.
        """
        url = f"https://coverartarchive.org/release-group/{mbid}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for image in data.get("images", []):
                    if image.get("front", False):
                        return image.get("image")  # Return the main image URL
                logging.warning(f"No front cover found for {mbid}")
                return None
            elif response.status_code == 404:
                logging.warning(f"No cover art found for {mbid} (404 Not Found)")
                return None
            else:
                logging.error(
                    f"Error fetching cover art ({response.status_code}): {response.text}"
                )
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching cover art for {mbid}: {e}")
            return None


class AlbumDataFrame:
    def __init__(self):
        self.df = pd.DataFrame(
            columns=[
                "mbid",
                "title",
                "artist",
                "genre_musicbrainz",
                "genre_discogs",
                "cover_url",
            ]
        )

    def add_album(self, album: Album):
        self.df = pd.concat(
            [self.df, pd.DataFrame([album.to_dict()])], ignore_index=True
        )

    def save_json(self, filename="albums.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.df.to_dict(orient="records"), f, indent=4)


if __name__ == "__main__":
    # Set up the logger
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting the evaluation process...")

    # Initialize clients
    musicbrainz = MusicBrainzClient()
    discogs = DiscogsClient(user_token="SjIfUoWCnPfVmqSEFxVaeAKxhRuYmHkmwsxqeqRv")
    cover_art = CoverArtClient()

    # Step 1: Fetch albums from MusicBrainz
    albums = musicbrainz.search_albums()

    # Step 2: Fetch genres from Discogs
    for album in albums:
        discogs.get_genre(album)

    # Step 3: Save to DataFrame and JSON
    df = AlbumDataFrame()
    for album in albums:
        df.add_album(album)

    df.save_json()
