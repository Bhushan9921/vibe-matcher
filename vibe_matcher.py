import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeit import default_timer as timer
import csv
from datetime import datetime

USE_OPENAI = False
OPENAI_MODEL = "text-embedding-3-small"
GOOD_THRESHOLD = 0.70

data = [
    {"name": "Boho Breeze Maxi Dress", "desc": "A flowy maxi dress with earthy tones, embroidery, and tassels—perfect for festivals and sunset picnics.", "vibes": ["boho", "free-spirited", "earthy", "festival"]},
    {"name": "City Sprint Sneakers", "desc": "Sleek low-profile sneakers with reflective accents—built for fast urban commutes and nightlife.", "vibes": ["urban", "energetic", "sporty", "streetwear"]},
    {"name": "Minimalist Monochrome Blazer", "desc": "Clean lines, matte finish, and sharp structure in black—elegant minimalism for modern offices.", "vibes": ["minimal", "monochrome", "chic", "office"]},
    {"name": "Cozy Knit Cardigan", "desc": "Oversized knit with soft texture and warm neutrals—designed for comfort, reading nooks, and coffee dates.", "vibes": ["cozy", "warm", "soft", "casual"]},
    {"name": "Tech Utility Cargo Pants", "desc": "Water-resistant cargo trousers with zip pockets and articulated knees—functional style for city explorers.", "vibes": ["utility", "techwear", "urban", "functional"]},
    {"name": "Pastel Street Hoodie", "desc": "Relaxed fit hoodie in washed pastel palette—youthful, playful, and made for weekend hangouts.", "vibes": ["streetwear", "playful", "casual", "color-pop"]},
    {"name": "Vintage Denim Jacket", "desc": "Faded indigo denim with contrast stitching—timeless throwback for concerts and road trips.", "vibes": ["vintage", "retro", "casual", "classic"]},
    {"name": "Scandi Wool Overcoat", "desc": "Structured wool overcoat with hidden placket and crisp silhouette—subtle luxury for winter days.", "vibes": ["scandi", "minimal", "luxury", "elegant"]},
    {"name": "Athleisure Air Leggings", "desc": "Breathable high-stretch leggings with mesh panels—gym-to-brunch comfort with a polished look.", "vibes": ["athleisure", "sporty", "comfortable", "polished"]},
    {"name": "Cottagecore Midi Skirt", "desc": "Soft floral midi with ruffle hem—romantic countryside mood for picnics and farm markets.", "vibes": ["cottagecore", "romantic", "soft", "floral"]}
]

df = pd.DataFrame(data)
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

def prepare_corpus(df):
    return (df["desc"] + " " + df["vibes"].apply(lambda v: " ".join(v))).tolist()

def embed_with_tfidf(texts):
    return tfidf_vectorizer.fit_transform(texts)

def embed_with_openai(texts):
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data])
    return vectors

corpus = prepare_corpus(df)
if USE_OPENAI:
    product_embeddings = embed_with_openai(corpus)
    backend = "openai"
else:
    product_embeddings = embed_with_tfidf(corpus)
    backend = "tfidf"

print(f"Embedding backend: {backend}")

def embed_query(query):
    if USE_OPENAI:
        return embed_with_openai([query])
    else:
        return tfidf_vectorizer.transform([query])

def vibe_fallback(query, df):
    tokens = set(query.lower().split())
    all_vibes = sorted({v for row in df["vibes"] for v in row})
    overlap_counts = [(v, len(tokens.intersection(set(v.split())))) for v in all_vibes]
    best = sorted(overlap_counts, key=lambda x: x[1], reverse=True)
    suggestion = best[0][0] if best and best[0][1] > 0 else all_vibes[0]
    return f"No strong match. Try searching a vibe like '{suggestion}' or broaden your phrase."

def search(query, top_k=3, threshold=0.25):
    q_vec = embed_query(query)
    sims = cosine_similarity(q_vec, product_embeddings).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "rank": len(results) + 1,
            "name": df.loc[i, "name"],
            "score": float(sims[i]),
            "desc": df.loc[i, "desc"],
            "vibes": ", ".join(df.loc[i, "vibes"])
        })
    meta = {"fallback": None}
    if len(results) == 0 or results[0]["score"] < threshold:
        meta["fallback"] = vibe_fallback(query, df)
    return results, meta

for q in ["energetic urban chic", "cozy cottagecore", "minimal monochrome"]:
    t0 = timer()
    res, meta = search(q, top_k=3)
    t1 = timer()
    print(f"\nQuery: {q}")
    for r in res:
        print(f"  {r['rank']}) {r['name']}  score={r['score']:.3f}  vibes=[{r['vibes']}]")
    if meta["fallback"]:
        print("  " + meta["fallback"])
    print(f"Latency: {t1 - t0:.4f}s")

log_file = "vibe_results.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "query", "top_match", "score", "vibes"])

print("\nType any vibe and press Enter to see top matches.")
print("All results will be saved to vibe_results.csv\n")

while True:
    user_query = input("Enter your vibe (or 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("\nAll your searches have been saved to vibe_results.csv.")
        break
    res, meta = search(user_query, top_k=3)
    print(f"\nTop matches for '{user_query}':")
    for r in res:
        print(f"  {r['rank']}) {r['name']}  score={r['score']:.3f}  vibes=[{r['vibes']}]")
    if meta["fallback"]:
        print(meta["fallback"])
    top = res[0]
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_query,
            top["name"],
            f"{top['score']:.3f}",
            top["vibes"]
        ])
