import pandas as pd

input_path = "data/raw/comments.csv"

df = pd.read_csv(input_path)

print("=" * 70)
print("COMMENT DATASET ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nComment text examples:")
print("-" * 70)

for i, text in enumerate(df["text"].head(10), start=1):
    print(f"{i}. {text}")

print("\nComment length statistics:")
print("-" * 70)

df["text"] = df["text"].fillna("").astype(str)

df["word_count"] = df["text"].str.split().str.len()
df["character_count"] = df["text"].str.len()

print(df[["word_count", "character_count"]].describe())

print("\n" + "=" * 70)
print("COMMENT ANALYSIS COMPLETE")
print("=" * 70)