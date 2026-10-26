# 🎯 Customer Segmentation Dashboard

An interactive Streamlit dashboard that groups customers into segments with **real K-Means clustering** (scikit-learn). No API keys, database, or paid services needed.

## Features
- Built-in sample data (`data/customers.csv`, or generated in code if the file is missing) plus CSV upload
- KPI cards: customers, average income, average spending score, clusters, silhouette score
- Pick numeric features, choose k from 2 to 10, features are scaled with `StandardScaler`
- Interactive scatter plot (chosen axes or PCA projection) with centroids
- Cluster sizes, average values, heatmap vs. overall average, and auto-written segment descriptions
- Elbow chart and silhouette-by-k chart to help choose k
- Download the dataset with a `Cluster` column
- Handles missing values (median fill or drop), ID/text columns, too few rows, and bad CSV files

## Run it
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Using your own CSV
Choose **Upload CSV** in the sidebar. You need at least two numeric columns (e.g. `Age`, `Annual Income`, `Spending Score`). ID columns and text columns are ignored automatically. Columns with "income" or "spend" in their name feed the KPI cards.

## Project structure
```
customer-segmentation/
├── app.py            # Streamlit UI + ML logic
├── requirements.txt
├── README.md
├── .gitignore
└── data/customers.csv
```

## How it works
1. Load and clean the data (numeric conversion, missing values)
2. Scale features with `StandardScaler`
3. Fit `KMeans(n_clusters=k, n_init=10)`
4. Score with silhouette, describe clusters by comparing each cluster's mean to the overall mean

## Scaling
- Up to 50,000 rows: exact `KMeans` (10 restarts)
- Above 50,000 rows: `MiniBatchKMeans` automatically (hundreds of thousands of rows in seconds)
- Silhouette and the elbow chart use random samples, so they stay fast
- The scatter plot draws at most 5,000 points (WebGL), but centroids, sizes and profiles use every row
- Results are cached, so moving between tabs does not re-run the model
- Use the **Sample size** slider in the sidebar to try 5k, 50k or 250k generated customers
- Upload limit is 500 MB (`.streamlit/config.toml`)

## Deploy
```bash
docker build -t customer-segmentation .
docker run -p 8501:8501 customer-segmentation
```
Or push the repo to GitHub and deploy it for free on Streamlit Community Cloud.
