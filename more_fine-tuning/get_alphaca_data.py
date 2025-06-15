import requests
import json

# MADE WITH CHATGPT

# URL of the JSON file
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# Download the JSON content
response = requests.get(url)
response.raise_for_status()  # Raises an error if the download fails

# Parse the JSON content
data = response.json()

# Save it locally
with open("training_data/alpaca_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved 'alpaca_data.json' successfully.")