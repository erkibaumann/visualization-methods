import requests
import json

# URL to fetch data from
url = "https://decision.cs.taltech.ee/electricity/api/"

# Fetch the data
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    try:
        # Parse the JSON response
        data = response.json()

        # List and enumerate dataset hashes
        print("Dataset Hashes:")
        for i, entry in enumerate(data, start=1):
            print(f"{i}. {entry['dataset']}")

        # Display different row values
        print("\nDataset Details:")
        for i, entry in enumerate(data, start=1):
            print(f"Dataset {i}:")
            print(f"  Filesize: {entry['filesize']} bytes")
            print(f"  Number of rows: {entry['num_rows']}")
            print(f"  Heated Area: {entry['heated_area']} m²")
            print(f"  Heat Source: {entry['heat_source']}")
            print("-" * 40)

    except json.JSONDecodeError:
        print("Error parsing JSON response")
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")