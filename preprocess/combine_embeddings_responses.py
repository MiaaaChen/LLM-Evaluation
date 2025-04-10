import os
import pandas as pd
import json

# Step 1: Combine all embeddings into a single CSV file
embeddings_directory = "embeddings"
all_embeddings = []

# Iterate over all CSV files in the directory to read embeddings
for filename in os.listdir(embeddings_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(embeddings_directory, filename)
        df = pd.read_csv(file_path, index_col=0)  # Assume the first column is the index

        # Extract the last two parts of the filename and form a unique identifier
        short_filename = "_".join(filename.split("_")[-2:]).replace(".csv", "")
        df.index = [f"{short_filename}_{idx}" for idx in df.index]

        all_embeddings.append(df)

# Combine all embeddings into a single DataFrame
combined_embeddings = pd.concat(all_embeddings, axis=0)

# Split the 'Embedding' column into multiple columns
if "Embedding" in combined_embeddings.columns:
    embeddings_split = combined_embeddings["Embedding"].str.split(",", expand=True)
    embeddings_split.columns = [
        f"Feature_{i+1}" for i in range(embeddings_split.shape[1])
    ]
    combined_embeddings = combined_embeddings.drop(columns=["Embedding"]).join(
        embeddings_split
    )

# Convert the combined embeddings to float
combined_embeddings = combined_embeddings.astype(float)

# Save the combined embeddings to a CSV file
combined_embeddings.to_csv("features_processed.csv", index=True)
print("features_processed.csv saved。")

# Step 2: Combine all responses into a single JSON file
responses_directory = "responses"
original_text_dict = {}

# Iterate over all CSV files in the directory to read responses
for filename in os.listdir(responses_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(responses_directory, filename)

        # Extract the last two parts of the filename and form a unique identifier
        short_filename = "_".join(filename.split("_")[-2:]).replace(".csv", "")

        # Read the CSV file and extract the 'Response' column
        df = pd.read_csv(file_path)

        # Iterate over each row in the DataFrame and save the response
        for _, row in df.iterrows():
            unique_index = f"{short_filename}_{row['Index']}"
            response_text = (
                row["Response"].replace("\n", " ").strip()
            )  # Remove newlines and leading/trailing spaces
            original_text_dict[unique_index] = response_text

# Save the original text dictionary to a JSON file
with open("original_text_dict.json", "w", encoding="utf-8") as f:
    json.dump(original_text_dict, f, ensure_ascii=False, indent=4)

print(f"original_text_dict.json saved with {len(original_text_dict)} data.")
