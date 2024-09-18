import pandas as pd
import csv
import argparse


def map_labels_to_mbid(csv_filename: str, original_dataframe: pd.DataFrame) -> pd.DataFrame:
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


def filter_most_frequent_genres(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Filter DataFrame to keep only the most frequent 'n' genres for each entry.

    Args:
        df (pd.DataFrame): DataFrame containing genre labels.
        top_n (int): The number of most frequent genres to keep.

    Returns:
        pd.DataFrame: DataFrame with only the most frequent 'n' genres for each entry.
    """
    genre_columns = df["genre_label"].str.split("/", expand=True)
    genre_counts = genre_columns.stack().value_counts()
    print(genre_counts)
    top_genres = genre_counts.head(top_n).index.tolist()
    print(top_genres)
    filtered_genre_columns = genre_columns.apply(
        lambda row: row[row.isin(top_genres)], axis=1
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process genre labels and MBID-image mappings.")
    parser.add_argument(
        "--original_tsv", required=True, help="Path to the original TSV file with genre labels."
    )
    parser.add_argument(
        "--mbid_csv", required=True, help="Path to the CSV file with MBID-image location mappings."
    )
    parser.add_argument(
        "--output_csv", required=True, help="Path to the output CSV file for mapped MBIDs and labels."
    )
    parser.add_argument(
        "--filtered_csv", required=True, help="Path to the output CSV file for filtered genres."
    )

    args = parser.parse_args()

    # Read the original dataframe
    original_dataframe = pd.read_csv(args.original_tsv, sep="\t")

    # Map labels to MBID
    mapped_dataframe = map_labels_to_mbid(args.mbid_csv, original_dataframe)
    if mapped_dataframe is not None:
        mapped_dataframe.to_csv(args.output_csv, index=False)

        # Filter most frequent genres
        filtered_dataframe = pd.read_csv(args.output_csv)
        filtered_dataframe = filter_most_frequent_genres(filtered_dataframe)
        filtered_dataframe.to_csv(args.filtered_csv, index=False)
