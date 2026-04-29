# BBC News NLP Pipeline — Subcategorization, NER & Event Extraction

A full unsupervised NLP pipeline built on the BBC News dataset that automatically subcategorizes articles, extracts named entities, and identifies April-related events across five news categories.

---

## Project Overview

The BBC News dataset contains hundreds of articles grouped into five broad categories: **business, entertainment, politics, sport, and tech**. While useful, these categories are too broad for meaningful analysis. This project builds an automated pipeline that breaks each category down into precise, semantically meaningful subcategories — with no manual labeling required.

In addition to subcategorization, the pipeline performs **Named Entity Recognition (NER)** and **April event extraction** on every article.

---

## Dataset

- **Source:** BBC News Dataset
- **Categories:** Business (439 articles), Entertainment, Politics, Sport, Tech (401 articles)
- **Format:** Raw `.txt` files, one article per file
- **Structure:** Five category folders, each processed by a dedicated notebook

---

## Pipeline Architecture

Each category has its own Jupyter notebook following the same pipeline:

### Phase 1 — Data Cleaning
- Encoding normalization
- URL and HTML stripping
- Special character removal
- Minimum word-length quality gate
- Stop words retained (transformer handles context internally)

### Phase 2 — Embedding (Jina AI API)
- **Model:** `jina-embeddings-v3-small`
- **Task:** `text-matching`
- **Dimensions:** 1024
- Embeddings saved to `.npy` files to avoid repeated API calls

### Phase 3 — Dimensionality Reduction (UMAP)
- `n_components=3`
- `n_neighbors=15`
- `min_dist=0.0`
- `metric='cosine'`
- `random_state=42`

### Phase 4 — Clustering (HDBSCAN)
- `min_cluster_size=8`
- `min_samples=4`
- `metric='euclidean'`
- `cluster_selection_method='eom'`
- Articles with no cluster assignment labeled as **Noise**

### Phase 5 — Cluster Labeling
- **c-TF-IDF** for cluster-level keyword extraction
- **KeyBERT** for semantic keyword extraction
- **Phase 5B:** Manual verification via sample titles and 80-word `cleaned_text` snippets per cluster

### Phase 6 — NER & April Event Extraction
- Named entity extraction: persons, organisations, locations, roles
- April event extraction: identifies events tied to April from article content

---

## Output Files

Each category produces two output CSVs:

### Subcategories Output (`{category}_subcategories_output.csv`)
| Column | Description |
|---|---|
| `file_name` | Source `.txt` file |
| `article_id` | Unique article identifier |
| `title` | Article title |
| `cleaned_text` | Preprocessed article body |
| `subcategory_id` | Cluster ID assigned by HDBSCAN |
| `semantic_label` | Human-readable subcategory name |
| `confidence_score` | HDBSCAN soft clustering confidence |
| `article_keywords` | KeyBERT-extracted keywords |

### NER Output (`{category}_ner_output.csv`)
| Column | Description |
|---|---|
| `named_persons` | Persons mentioned in article |
| `named_organisations` | Organisations mentioned |
| `named_locations` | Locations mentioned |
| `person_roles` | Roles associated with persons |
| `april_events` | April-related events extracted |

---

## Business Category Results

- **Articles processed:** 439
- **Subcategories discovered:** 18+
- **Average confidence score:** 0.86

**Confirmed subcategories include:**
Automotive Industry, Energy Markets, Housing Market, Pharmaceuticals, Natural Disaster Economics, Crude Oil Markets, Asian Economics, Pension Funds, Macroeconomic Indicators, International Trade, Foreign Exchange, Food Industry, Corporate Governance, Aviation Industry, Telecommunications, Financial Markets, Sports Business, Beverages Industry, Media Industry, Retail Industry

---

## Notebooks

| Notebook | Category |
|---|---|
| `business_subcategories.ipynb` | Business |
| `entertainment_subcategories.ipynb` | Entertainment |
| `politics_subcategories.ipynb` | Politics |
| `sport_subcategories.ipynb` | Sport |
| `tech_subcategories.ipynb` | Tech |
| `bbc_nlp_recommendation.ipynb` | Recommendation Engine |

---

## How to Run

> ⚠️ Run cells **one by one** — never use "Run All". The embedding and clustering cells save outputs to `.npy` files to prevent redundant API calls. Re-running them will trigger new API requests and may alter cluster assignments.

1. Clone the repository
2. Install dependencies (see below)
3. Add your Jina AI API key where required
4. Open the notebook for your target category
5. Run cells sequentially from top to bottom

---

## Dependencies

```
numpy
pandas
umap-learn
hdbscan
keybert
scikit-learn
requests
jupyter
```

Install with:
```bash
pip install numpy pandas umap-learn hdbscan keybert scikit-learn requests jupyter
```

---

## Project Structure

```
BBC_NLP_PROJECT/
│
├── business_subcategories.ipynb
├── entertainment_subcategories.ipynb
├── politics_subcategories.ipynb
├── sport_subcategories.ipynb
├── tech_subcategories.ipynb
├── bbc_nlp_recommendation.ipynb
│
├── business_subcategories_output.csv
├── business_ner_output.csv
├── entertainment_subcategories_output.csv
├── entertainment_ner_output.csv
│   ... (and so on for each category)
│
├── politics/        ← raw .txt article files
├── sport/
├── tech/
│
└── .gitignore
```

---

## Key Design Decisions

- **Why HDBSCAN over KMeans?** HDBSCAN does not require specifying the number of clusters upfront and handles noise naturally — critical for open-ended topic discovery.
- **Why Jina embeddings over bag-of-words?** Transformer embeddings capture semantic meaning, not just word frequency. Articles about similar topics cluster together even when they use different vocabulary.
- **Why save `.npy` files?** Embedding 400+ articles via API is expensive and slow. Saving outputs means the pipeline can be resumed at any stage without re-running earlier steps.
- **Why no stop word removal?** The transformer model handles context internally — removing stop words is unnecessary and can hurt semantic quality.

---

## Author

**Jayeoba Timileyin**
NLP & Data Science | Obafemi Awolowo University (OAU), Ife

📧 timileyinjayeoba@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/timileyin-emmanuel-17616b37a)