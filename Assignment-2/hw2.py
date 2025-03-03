import pandas as pd
import numpy as np
import krippendorff
import matplotlib.pyplot as plt

# Load the dataset
file_path = "multidoGo_all_new.csv"

# Read the CSV
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

# Get all unique categorical tag combinations across annotators
unique_tag_combinations = list(set(df["kiara_tags"]) | set(df["jack_tags"]) | set(df["shriya_tags"]))

# Create a mapping from tag combinations to categorical indices
tag_combination_to_index = {tag_comb: i for i, tag_comb in enumerate(unique_tag_combinations)}

# Use .apply() instead of .map() to avoid reindexing errors
df["kiara_index"] = df["kiara_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))
df["jack_index"] = df["jack_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))
df["shriya_index"] = df["shriya_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))

# Stack coder annotations into a 2D matrix (rows = coders, columns = items)
annotation_matrix = np.array([
    df["kiara_index"].values,
    df["jack_index"].values,
    df["shriya_index"].values
])

print("Fixed Annotation matrix shape:", annotation_matrix.shape)  # Should be (3, num_items)

# Compute Krippendorff's alpha
alpha = krippendorff.alpha(annotation_matrix, level_of_measurement="nominal")

# Calculate agreement percentage per coder pair
def compute_agreement(df, col1, col2):
    """Compute agreement percentage between two columns"""
    return np.mean(df[col1] == df[col2])

kiara_jack_agreement = compute_agreement(df, "kiara_index", "jack_index")
kiara_shriya_agreement = compute_agreement(df, "kiara_index", "shriya_index")
jack_shriya_agreement = compute_agreement(df, "jack_index", "shriya_index")

# Store agreement results in a dataframe
results_df = pd.DataFrame({
    "Coder Pair": ["Kiara & Jack", "Kiara & Shriya", "Jack & Shriya"],
    "Agreement": [kiara_jack_agreement, kiara_shriya_agreement, jack_shriya_agreement]
})

# Display results
print("Coder Agreement Results:")
print(results_df)

# Plot agreement distribution
plt.figure(figsize=(8, 6))
plt.bar(results_df["Coder Pair"], results_df["Agreement"], color="skyblue")
plt.xlabel("Coder Pairs")
plt.ylabel("Agreement Percentage")
plt.title("Inter-Annotator Agreement between Coders")
plt.ylim(0, 1)
plt.show()

# Output Krippendorff's Alpha
print(f"Krippendorff’s Alpha: {alpha:.3f}")


import pandas as pd
import numpy as np
import krippendorff
import matplotlib.pyplot as plt

# Load the dataset
file_path = "ubuntu_all_new.csv"

# Read the CSV
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

# Get all unique categorical tag combinations across annotators
unique_tag_combinations = list(set(df["kiara_tags"]) | set(df["jack_tags"]) | set(df["shriya_tags"]))

# Create a mapping from tag combinations to categorical indices
tag_combination_to_index = {tag_comb: i for i, tag_comb in enumerate(unique_tag_combinations)}

# Use .apply() instead of .map() to avoid reindexing errors
df["kiara_index"] = df["kiara_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))
df["jack_index"] = df["jack_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))
df["shriya_index"] = df["shriya_tags"].apply(lambda x: tag_combination_to_index.get(x, -1))

# Stack coder annotations into a 2D matrix (rows = coders, columns = items)
annotation_matrix = np.array([
    df["kiara_index"].values,
    df["jack_index"].values,
    df["shriya_index"].values
])

print("\n\n\n\nFixed Annotation matrix shape:", annotation_matrix.shape)  # Should be (3, num_items)

# Compute Krippendorff's alpha
alpha = krippendorff.alpha(annotation_matrix, level_of_measurement="nominal")

# Calculate agreement percentage per coder pair
def compute_agreement(df, col1, col2):
    """Compute agreement percentage between two columns"""
    return np.mean(df[col1] == df[col2])

kiara_jack_agreement = compute_agreement(df, "kiara_index", "jack_index")
kiara_shriya_agreement = compute_agreement(df, "kiara_index", "shriya_index")
jack_shriya_agreement = compute_agreement(df, "jack_index", "shriya_index")

# Store agreement results in a dataframe
results_df = pd.DataFrame({
    "Coder Pair": ["Kiara & Jack", "Kiara & Shriya", "Jack & Shriya"],
    "Agreement": [kiara_jack_agreement, kiara_shriya_agreement, jack_shriya_agreement]
})

# Display results
print("Coder Agreement Results:")
print(results_df)

# Plot agreement distribution
plt.figure(figsize=(8, 6))
plt.bar(results_df["Coder Pair"], results_df["Agreement"], color="skyblue")
plt.xlabel("Coder Pairs")
plt.ylabel("Agreement Percentage")
plt.title("Inter-Annotator Agreement between Coders")
plt.ylim(0, 1)
plt.show()

# Output Krippendorff's Alpha
print(f"Krippendorff’s Alpha: {alpha:.3f}")