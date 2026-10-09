# movie_recommender_system

A complete, end-to-end Machine Learning project that handles data preprocessing, text vectorization, and similarity modeling, served over an interactive user dashboard. The engine analyzes semantic overlap in movie plot overviews and genres to surface highly relevant recommendations paired with dynamic, live-fetched artwork.

---

## 🚀 Features

*   **Semantic Text Vectorization:** Combines structural movie descriptions and genres into unified semantic profiles.
*   **Mathematical Similarity Mapping:** Utilizes high-dimensional Cosine Similarity vectors to rank items.
*   **Dynamic Poster Fetching:** Leverages TMDB API integrations to load high-resolution posters asynchronously during inference.
*   **Interactive Web UI:** Features a custom frontend carousel element and drop-down selectors built natively with Streamlit.

---

## 🛠️ Tech Stack & Core Libraries

*   **Language:** Python 3.8+
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (`CountVectorizer`, `cosine_similarity`)
*   **Serialization:** Pickle (Binary protocol archiving)
*   **Dashboard Framework:** Streamlit, Streamlit Components v1
*   **Network & Integration:** Requests API (TMDB live-endpoint synchronization)

---

## 📐 Machine Learning Pipeline Architecture

### 1. Feature Engineering & Preprocessing
*   The raw movie metadata (`overview` + `genre`) is tokenized and merged into a compound textual metadata string named `tags`.
*   Missing text vectors are handled, and values are cast uniformly into UTF-8 formats.

### 2. Vectorization & Vector Space Modeling
*   A `CountVectorizer` parses the corpus token strings to retain a maximum of `10,000` dominant text features while automatically dropping english stop words.
*   The raw text fields are projected into a sparse, localized token matrix array (10,000 × 10,000).

### 3. Similarity Formulation
*   A pairwise **Cosine Similarity matrix** computes the angular distances between all multi-dimensional movie arrays:
\[\text{Similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}\]
*   The resulting distance weights are preserved inside a high-speed matrix framework for real-time drop-down sorting.

---

## 🛣️ Inference Pipeline (The Streamlit Application)

When a user selects a target title from the system dropdown menu:
1. The app identifies the internal dataset index corresponding to the title string.
2. It fetches the target distance array slice from the pre-computed similarity matrix.
3. The indices are enumerated, sorted inversely by distance coefficients, and the top 5 closest metadata items (excluding the source movie itself) are returned.
4. The system triggers parallel HTTP GET requests using the movie's unique identifier to `api.themoviedb.org` to unpack the asset `poster_path`.

---

## 🛠️ Installation & Reproduction Setup

### Prerequisites
* Get a free API Key from [The Movie Database (TMDB)](https://themoviedb.org) to populate your poster pipeline.

### Step 1: Environment Configuration
```bash
# Clone the repository
git clone https://github.com
cd movie-recommender-streamlit

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install strictly isolated dependencies
pip install pandas scikit-learn requests streamlit
```

### Step 2: Model Artifact Preparation
Ensure you run your exploratory data script (`movie_recommender.ipynb` or your modeling script) first to generate your target pickle data structures:
```bash
# Ensure these binaries exist in your root folder before launching the app
movies_list.pkl
similarity.pkl
```

### Step 3: Run the Dashboard Application
```bash
streamlit run app.py
```

---

## 📁 Repository Layout
```text
├── frontend/              # Public asset folder maps for the embedded image carousel
│   └── public/            
├── app.py                 # Core Streamlit app containing UI layouts & TMDB poster fetching
├── movie_notebook.ipynb   # Jupyter Notebook tracking feature parsing & similarity computations
├── movies_list.pkl        # Compressed serialization dictionary for DataFrame assets
├── similarity.pkl         # Heavy matrix serialization file containing similarity distances
└── README.md              
```
