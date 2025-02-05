import argparse
import pandas as pd


def process_tsv(input_file, output_file):
    # Read the TSV file
    df = pd.read_csv(input_file, sep="\t")

    # Remove duplicates based on 'album_index'
    df_unique = df.drop_duplicates(subset="album_index")

    # Save the resulting dataframe to a CSV file
    df_unique.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process a TSV file to keep unique album_index and convert to CSV."
    )
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file")

    # Parse arguments
    args = parser.parse_args()

    # Process the TSV and create a CSV
    process_tsv(args.input_file, args.output_file)

    print(f"Processed file saved to {args.output_file}")
