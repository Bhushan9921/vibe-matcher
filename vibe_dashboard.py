import pandas as pd
import matplotlib.pyplot as plt
import os

log_file = "vibe_results.csv"

if not os.path.exists(log_file):
    print("No results found. Run vibe_matcher.py first.")
    exit()

df = pd.read_csv(log_file)

print("\n========= VIBE ANALYTICS DASHBOARD =========")
print(f"Total queries logged: {len(df)}")
print(f"Unique products matched: {df['top_match'].nunique()}")
print(f"Average similarity score: {df['score'].astype(float).mean():.3f}")
print("============================================\n")

top_products = df["top_match"].value_counts().head(5)
plt.figure(figsize=(6,4))
top_products.plot(kind="bar")
plt.title("Top 5 Most Matched Products")
plt.xlabel("Product")
plt.ylabel("Times Matched")
plt.tight_layout()
plt.show()

scores = df["score"].astype(float)
plt.figure(figsize=(6,4))
plt.hist(scores, bins=10, edgecolor="black")
plt.title("Distribution of Similarity Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
