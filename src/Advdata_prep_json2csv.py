import json
import csv
from pathlib import Path
import code ## code.interact(local=locals())

DOWNLOAD_DIRECTORY = "/Users/paulj/Documents/Github/PLM-ICD/data"
download_dir = Path(DOWNLOAD_DIRECTORY)
json_f = download_dir / "advdata.json"

def json_to_csv(json_file_path, csv_file_path):
    """
    Converts a JSON file to a CSV file.

    Each key in the JSON object will become a column in the CSV.

    :param json_file_path: Path to the input JSON file
    :param csv_file_path: Path to the output CSV file
    """
    try:
        # Open and load the JSON file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Ensure the data is a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of objects.")

        # Process the "text" key for each entry
        for entry in data:
            if "text" in entry and isinstance(entry["text"], dict):
                entry["text"] = " ".join(str(value) for value in entry["text"].values())

        # Get the headers from the keys of the first dictionary
        headers = data[0].keys()
        # Write to CSV
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ['primary',"secondary", 'text'], extrasaction='ignore')

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for row in data:
                writer.writerow(row)

        print(f"Successfully converted {json_file_path} to {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
json_to_csv(json_f, download_dir / 'json_output.csv')


