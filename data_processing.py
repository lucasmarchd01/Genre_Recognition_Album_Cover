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
    # Split the genre labels into separate columns
    genre_columns = df["genre_label"].str.split("/", expand=True)

    # Stack the genre columns to a single column and count the occurrences
    genre_counts = genre_columns.stack().value_counts()

    # Select the top 5 most frequent genres
    top_5_genres = genre_counts.head(7).index.tolist()

    # Filter the genre columns to keep only the top 5 most frequent genres
    filtered_genre_columns = genre_columns.apply(
        lambda row: row[row.isin(top_5_genres)], axis=1
    )

    # Join the top 5 most frequent genres back into a single column
    df["genre_label"] = filtered_genre_columns.apply(
        lambda row: "/".join(row.dropna()), axis=1
    )

    # Remove rows where all genre labels have been removed
    df = df[df["genre_label"] != ""]

    return df


def add_tilde_slash(image_csv, new_image_csv):
    with open(image_csv, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    for row in data:
        row[1] = "~/{}".format(row[1])

    with open(new_image_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


# # Example usage:
# original_image_csv = "data/csv/mapped_mbid_with_labels.csv"
# new_image_csv = "data/csv/final.csv"
# add_tilde_slash(original_image_csv, new_image_csv)


# original_dataframe = pd.read_csv(
#     "/Users/lucasmarch/Projects/AcousticBrainz/acousticbrainz-mediaeval-allmusic-train.tsv",
#     sep="\t",
# )
# mapped_dataframe = map_labels_to_mbid("mbid_to_image_filenames.csv", original_dataframe)

# output_filename = "mapped_mbid_with_labels.csv"
# mapped_dataframe.to_csv(output_filename, index=False)


mapped_dataframe = pd.read_csv("data/csv/final.csv")
filtered_dataframe = filter_most_frequent_genres(mapped_dataframe)
filtered_dataframe.to_csv("data/csv/final_top_6.csv", index=False)
