import pandas as pd
import numpy as np

# Function to load and preprocess dataset
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Normalize text: Convert to lowercase and strip whitespace
    for col in ["kiara", "jack", "shriya"]:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    # Convert multiple tags into a tuple-based categorical representation
    def parse_tags(tag_string):
        """Split and normalize multi-tag strings, convert to a sorted tuple for categorical encoding."""
        return tuple(sorted(tag_string.split(","))) if isinstance(tag_string, str) else tuple()

    df["kiara_tags"] = df["kiara"].apply(parse_tags)
    df["jack_tags"] = df["jack"].apply(parse_tags)
    df["shriya_tags"] = df["shriya"].apply(parse_tags)

    return df

# Function to find examples where all tags are the same or different
def find_examples(df):
    same_tags = df[(df["kiara_tags"] == df["jack_tags"]) & (df["jack_tags"] == df["shriya_tags"])]
    different_tags = df[(df["kiara_tags"] != df["jack_tags"]) & (df["jack_tags"] != df["shriya_tags"]) & (df["kiara_tags"] != df["shriya_tags"])]
    
    # Ensure "utterance" exists before selecting it
    columns_to_display = ["kiara", "jack", "shriya", "utterance"] if "utterance" in df.columns else ["kiara", "jack", "shriya"]
    
    # Select three examples for each case
    same_examples = same_tags[columns_to_display].head(3).copy()
    different_examples = different_tags[columns_to_display].head(3).copy()

    return same_examples, different_examples

# Process MultiDoGo dataset
multidogo_file = "multidoGo_all.csv"
multidogo_df = load_and_preprocess(multidogo_file)
multidogo_same, multidogo_different = find_examples(multidogo_df)

# Process Ubuntu dataset
ubuntu_file = "ubuntu_all.csv"
ubuntu_df = load_and_preprocess(ubuntu_file)
ubuntu_same, ubuntu_different = find_examples(ubuntu_df)

# Print results for MultiDoGo dataset
print("\n===== MultiDoGo Dataset =====")
print("\nExamples where all annotators agree (MultiDoGo):")
print(multidogo_same.to_string(index=False))

print("\nExamples where all annotators disagree (MultiDoGo):")
print(multidogo_different.to_string(index=False))

# Print results for Ubuntu dataset
print("\n===== Ubuntu Dataset =====")
print("\nExamples where all annotators agree (Ubuntu):")
print(ubuntu_same.to_string(index=False))

print("\nExamples where all annotators disagree (Ubuntu):")
print(ubuntu_different.to_string(index=False))