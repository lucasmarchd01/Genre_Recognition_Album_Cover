import pandas as pd
import csv


def map_labels_to_mbid(
    csv_filename: str, original_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Map genre labels to MBIDs from the generated CSV file.

    Args:
        csv_filename (str): The filename of the CSV file containing MBID-image location mappings.
        original_dataframe (pd.DataFrame): The original dataframe containing genre labels.

    Returns:
        pd.DataFrame: A dataframe containing MBIDs, image locations, and corresponding genre labels.
    """
    try:
        df_mapping = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"CSV file '{csv_filename}' not found.")
        return None

    # Merge the mapping dataframe with the original dataframe on MBID
    merged_df = pd.merge(
        df_mapping,
        original_dataframe,
        left_on="mbid",
        right_on="releasegroupmbid",
        how="left",
    )

    # Select relevant columns and rename them
    merged_df = merged_df[["mbid", "image_location", "genre1"]]
    merged_df.rename(columns={"genre1": "genre_label"}, inplace=True)
    merged_df.drop_duplicates(subset=["mbid"], inplace=True)

    return merged_df


def filter_most_frequent_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to keep only the most frequent 5 genres for each entry.

    Args:
        df (pd.DataFrame): DataFrame containing genre labels.

    Returns:
        pd.DataFrame: DataFrame with only the most frequent 5 genres for each entry.
    """
    genre_columns = df["genre_label"].str.split("/", expand=True)
    genre_counts = genre_columns.stack().value_counts()
    top_5_genres = genre_counts.head(7).index.tolist()
    filtered_genre_columns = genre_columns.apply(
        lambda row: row[row.isin(top_5_genres)], axis=1
    )
    df["genre_label"] = filtered_genre_columns.apply(
        lambda row: "/".join(row.dropna()), axis=1
    )
    df = df[df["genre_label"] != ""]

    return df


def remove_tilde_slash(image_csv, new_image_csv):
    with open(image_csv, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    for row in data:
        row[1] = row[1].replace("~/", "")

    with open(new_image_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


original_dataframe = pd.read_csv(
    "acousticbrainz-mediaeval-allmusic-train.tsv",
    sep="\t",
)
mapped_dataframe = map_labels_to_mbid("mbid_to_image_filenames.csv", original_dataframe)

output_filename = "mapped_mbid_with_labels.csv"
mapped_dataframe.to_csv(output_filename, index=False)


mapped_dataframe = pd.read_csv("data/csv/mapped_mbid_with_labels.csv")
filtered_dataframe = filter_most_frequent_genres(mapped_dataframe)
filtered_dataframe.to_csv("data/csv/final_top_6.csv", index=False)
