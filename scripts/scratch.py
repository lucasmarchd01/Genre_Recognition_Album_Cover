import json
import csv

# File paths
json_input_filename = "/Users/lucasmarch/Projects/Genre_Recognition_Album_Cover/results/llm/results_3123bb4d-05bf-41fa-9a1f-e8183b4ed67a/results_3123bb4d-05bf-41fa-9a1f-e8183b4ed67a.json"
csv_input_filename = (
    "/Users/lucasmarch/Projects/Genre_Recognition_Album_Cover/new_releaves.csv"
)
json_output_filename = "filtered_results_output.json"
csv_output_filename = "filtered_csv_new_releases_output.csv"

# Load the JSON file
with open(json_input_filename, "r", encoding="utf-8") as file:
    json_data = json.load(file)

genre_list = json_data.get("test", [])

# Load the CSV file
with open(csv_input_filename, "r", encoding="utf-8") as file:
    reader = list(csv.reader(file))
    header = reader[0]
    rows = reader[1:]

# Filter data
filtered_genres = []
filtered_rows = []

for genre, row in zip(genre_list, rows):
    if genre and genre.lower() != "error":  # Skip empty or "error" genres
        filtered_genres.append(genre)
        filtered_rows.append(row)

# Save the filtered JSON file
with open(json_output_filename, "w", encoding="utf-8") as file:
    json.dump({"test": filtered_genres}, file, indent=4)

# Save the filtered CSV file
with open(csv_output_filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(filtered_rows)

print(f"Filtered JSON saved to {json_output_filename}")
print(f"Filtered CSV saved to {csv_output_filename}")
