# Movie Recommendation System Using Cosine Similarity

## 📋 Project Overview
This project implements a **Content-Based Movie Recommendation System** using **Cosine Similarity** and **TF-IDF (Term Frequency-Inverse Document Frequency)** text vectorization. Given a movie name, the system recommends similar movies based on genres, keywords, taglines, cast, and director information.

## 🎯 Problem Statement
Build an intelligent recommendation system that suggests movies similar to a user's favorite movie. This helps users discover new content that matches their preferences based on movie attributes rather than just popularity or user ratings.

## 📊 Dataset
**File:** `movies.csv`

**Key Features:**
- `title` - Movie name (primary identifier)
- `genres` - Movie genres (e.g., Action, Drama, Sci-Fi)
- `keywords` - Descriptive keywords about the movie
- `tagline` - Movie tagline/slogan
- `cast` - Main actors/actresses
- `director` - Movie director
- `index` - Unique movie identifier

**Dataset Size:** ~4,800 movies

---

## 🧠 Theory: Cosine Similarity & Content-Based Filtering

### What is Content-Based Filtering?
**Content-Based Filtering** recommends items similar to those a user liked in the past, based on item features/attributes. Unlike collaborative filtering (which uses user behavior), content-based filtering focuses on item characteristics.

**Example:**
- User likes "Iron Man" → System recommends other superhero/action movies with similar cast/director

### Cosine Similarity

#### Mathematical Definition
**Cosine Similarity** measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. It determines how similar two vectors are, regardless of their magnitude.

**Formula:**
$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Where:
- $A \cdot B$ = Dot product of vectors A and B
- $\|A\|$ = Magnitude (length) of vector A
- $\|B\|$ = Magnitude (length) of vector B

#### Range and Interpretation
- **Range:** -1 to 1 (for normalized vectors: 0 to 1)
- **1 or 100%:** Vectors point in the same direction (identical)
- **0:** Vectors are orthogonal (no similarity)
- **-1:** Vectors point in opposite directions (for text, usually not negative)

#### Why Cosine Similarity for Text?

1. **Magnitude Independence:** Focuses on direction, not length
   - "action movie superhero" vs "action action movie superhero superhero"
   - Same meaning, different word counts → High similarity

2. **High-Dimensional Data:** Works well with sparse TF-IDF vectors
3. **Normalized Comparison:** Fair comparison regardless of text length
4. **Efficient Computation:** Fast with optimized libraries

#### Visual Intuition
```
Vector A: Iron Man     → [0.8, 0.6, 0.2, ...]
Vector B: Avengers     → [0.7, 0.7, 0.1, ...]
Vector C: Titanic      → [0.1, 0.2, 0.9, ...]

Cosine(A, B) = 0.95 (very similar - both superhero movies)
Cosine(A, C) = 0.23 (not similar - different genres)
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

#### What is TF-IDF?
**TF-IDF** is a numerical statistic that reflects how important a word is to a document in a collection of documents. It's used to convert text into numerical feature vectors.

#### Components

**1. Term Frequency (TF):**
$$TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

**Intuition:** How frequently does a word appear in this specific document?

**2. Inverse Document Frequency (IDF):**
$$IDF(t, D) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right)$$

**Intuition:** How rare is this word across all documents?

**3. TF-IDF Score:**
$$\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)$$

#### Why TF-IDF?

**Problem with Raw Word Counts:**
- Common words like "the", "is", "a" dominate
- Important but rare words get underweighted

**TF-IDF Solution:**
- **High TF-IDF:** Word appears frequently in this document but rarely in others (important!)
- **Low TF-IDF:** Either rare in this document OR common across all documents (less important)

#### Example
```
Document 1: "action superhero action movie"
Document 2: "romantic movie love story"

For "action" in Doc 1:
TF = 2/4 = 0.5
IDF = log(2/1) = 0.3
TF-IDF = 0.5 × 0.3 = 0.15

For "movie" in Doc 1:
TF = 1/4 = 0.25
IDF = log(2/2) = 0  (appears in both docs)
TF-IDF = 0.25 × 0 = 0  (not discriminative)
```

### difflib - Close Match Finding

**Purpose:** Handle typos and variations in movie names

```python
difflib.get_close_matches("thor", list_of_movies)
# Returns: ["Thor", "Thor: The Dark World", "Thor: Ragnarok"]
```

**Algorithm:** Uses **SequenceMatcher** based on:
- Longest common substring
- Edit distance (Levenshtein distance)
- Similarity ratio threshold

**Why Needed?**
- User input: "iron man" → Database: "Iron Man"
- User input: "avengrs" → Finds: "Avengers"
- Improves user experience with fuzzy matching

---

## 🔧 Implementation Details

### 1. Data Loading and Exploration
```python
df = pd.read_csv("../../dataset/movies.csv")
df.head()
df.shape  # (~4800, columns)
```

### 2. Feature Selection
```python
features = ['genres', 'keywords', 'tagline', 'cast', 'director']
```

**Rationale:**
- **Genres:** Core movie category (Action, Drama)
- **Keywords:** Plot elements, themes
- **Tagline:** Essence of the movie
- **Cast:** Actor similarities
- **Director:** Directorial style similarities

**Not Used:** Plot summary (too long, noisy)

### 3. Data Cleaning
```python
for feature in features:
    df[feature] = df[feature].fillna('')
```

**Why Fill with Empty String?**
- Some movies have missing cast/tagline
- Empty string doesn't affect TF-IDF
- Prevents NaN errors in string concatenation

### 4. Feature Combination
```python
combined_features = (df['genres'] + ' ' + 
                    df['keywords'] + ' ' + 
                    df['tagline'] + ' ' + 
                    df['cast'] + ' ' + 
                    df['director'])
```

**Result:** Single text representation per movie
```
Example: "Action Sci-Fi superhero technology genius 
          Iron Man suit Tony Stark Robert Downey Jr Jon Favreau"
```

### 5. Text Vectorization
```python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
```

**Output:**
- Sparse matrix: (4800 movies, ~20,000 unique words)
- Each movie → numerical vector
- Each dimension → one unique word's TF-IDF score

**Example Matrix Structure:**
```
         action  superhero  romance  drama  ...
Movie1   0.85    0.72       0.00     0.12   ...
Movie2   0.91    0.68       0.00     0.05   ...
Movie3   0.00    0.00       0.89     0.76   ...
```

### 6. Similarity Computation
```python
similarity = cosine_similarity(feature_vectors)
```

**Output:**
- Square matrix: (4800 × 4800)
- `similarity[i][j]` = similarity between movie i and movie j
- Diagonal values = 1.0 (movie with itself)

**Visualization:**
```
        Movie1  Movie2  Movie3  Movie4
Movie1  1.00    0.85    0.12    0.45
Movie2  0.85    1.00    0.08    0.52
Movie3  0.12    0.08    1.00    0.34
Movie4  0.45    0.52    0.34    1.00
```

### 7. Recommendation Function

#### Step 1: Find Close Match
```python
movie_name = "thor"
list_of_movies = df['title'].tolist()
close_matches = difflib.get_close_matches(movie_name, list_of_movies)
closest_match = close_matches[0]  # "Thor"
```

#### Step 2: Get Movie Index
```python
movie_index = df[df.title == closest_match]['index'].values[0]
```

#### Step 3: Get Similarity Scores
```python
similarity_scores = list(enumerate(similarity[movie_index]))
# [(0, 0.12), (1, 0.85), (2, 1.0), (3, 0.45), ...]
# (movie_index, similarity_score)
```

#### Step 4: Sort by Similarity
```python
sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
# Highest similarity first
```

#### Step 5: Display Recommendations
```python
print("Movies suggested for you:\n")
for i, movie in enumerate(sorted_movies[1:31], 1):  # Skip first (itself)
    movie_index = movie[0]
    title = df[df.index == movie_index]['title'].values[0]
    print(f"{i}. {title}")
```

**Why Skip First?**
- First movie is always the input movie itself (similarity = 1.0)

---

## 📈 Results

### Example 1: "Thor"
**Recommendations:**
1. Thor: The Dark World
2. Thor: Ragnarok
3. Avengers
4. Captain America
5. Iron Man
6. Guardians of the Galaxy
... (30 total)

**Analysis:** System successfully recommends Marvel superhero movies!

### Example 2: "Iron Man"
**Recommendations:**
1. Iron Man 2
2. Iron Man 3
3. Avengers: Age of Ultron
4. Captain America: Civil War
5. The Avengers
... (20 total)

**Analysis:** Strong focus on MCU movies with shared cast/universe

---

## 💻 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib difflib scikit-learn
```

### Execution Steps
1. Ensure dataset is in `../../dataset/movies.csv`
2. Open `MovieRecomendation.ipynb` in Jupyter Notebook
3. Run cells 1-10 to train the model
4. In the last cells, change `movie_name` variable:
   ```python
   movie_name = "your favorite movie"
   ```
5. Run recommendation cells to get suggestions

### Interactive Use
```python
# Modify this cell for different movies
movie_name = "interstellar"  # Try: "titanic", "avatar", "joker"
```

---

## 🗂️ Project Structure
```
cosiine_similarity/
├── MovieRecomendation.ipynb    # Main implementation
└── README.md                   # This file
```

---

## 📚 Key Learnings

1. **Content-Based Filtering:** Recommends based on item attributes, not user behavior
2. **TF-IDF Vectorization:** Converts text to meaningful numerical representations
3. **Cosine Similarity:** Effective metric for comparing high-dimensional vectors
4. **Feature Engineering:** Combining multiple text features improves recommendations
5. **Fuzzy Matching:** Handle user input variations gracefully
6. **Scalability:** Pre-computing similarity matrix enables fast lookups

---

## 🔍 Algorithm Complexity

### Time Complexity
- **Vectorization:** O(n × m) where n = movies, m = avg words per movie
- **Similarity Matrix:** O(n² × m) - Most expensive step
- **Recommendation:** O(n log n) for sorting

### Space Complexity
- **Feature Vectors:** O(n × v) where v = vocabulary size
- **Similarity Matrix:** O(n²) - Can be memory-intensive for large datasets

### Optimization for Large Datasets
- Use approximate nearest neighbors (ANN)
- Pre-filter by genre before computing similarity
- Implement caching for popular movies

---

## 🔮 Future Improvements

### Algorithm Enhancements
1. **Hybrid Approach:** Combine content-based + collaborative filtering
2. **Weighted Features:** Give more importance to genres/director
3. **Word Embeddings:** Use Word2Vec or BERT instead of TF-IDF
4. **Sentiment Analysis:** Consider review sentiments in taglines
5. **User Personalization:** Learn user preferences over time

### Technical Improvements
1. **Database Integration:** Use SQL/NoSQL for movie storage
2. **API Development:** Build REST API for recommendations
3. **Web Interface:** Create user-friendly web app
4. **Real-time Updates:** Handle new movies dynamically
5. **Performance:** Implement approximate nearest neighbors

### Feature Additions
1. **Movie Ratings:** Incorporate IMDb/TMDB ratings
2. **Release Year:** Consider temporal preferences
3. **Language/Country:** Filter by user preferences
4. **Duration:** Recommend similar-length movies
5. **Mood-based:** "Show me uplifting movies like..."

### Evaluation Metrics
1. **Precision@K:** How many top-K recommendations are relevant?
2. **Diversity:** Are recommendations varied or repetitive?
3. **User Feedback:** Thumbs up/down for recommendations
4. **A/B Testing:** Compare with other algorithms

---

## 🎯 Strengths and Weaknesses

### Strengths ✅
- **No Cold Start:** Works immediately without user history
- **Transparency:** Recommendations are explainable
- **Privacy-Friendly:** Doesn't need personal data
- **Niche Content:** Can recommend less popular movies
- **Independence:** Doesn't need other users' data

### Weaknesses ❌
- **Limited Serendipity:** Only recommends similar movies (no surprises)
- **Feature Engineering:** Requires good domain knowledge
- **New User Problem:** Needs initial preference input
- **Overspecialization:** May create "filter bubble"
- **Content Dependency:** Quality depends on metadata quality

---

## 📖 Key Concepts Summary

### Cosine Similarity
- Measures angle between vectors
- Range: 0 to 1 (for positive vectors)
- Higher = more similar

### TF-IDF
- Converts text to numbers
- Highlights important, discriminative words
- Downweights common words

### Content-Based Filtering
- Uses item features
- Independent of other users
- Requires good metadata

---

## 📖 References

- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Cosine Similarity Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [Recommendation Systems Guide](https://developers.google.com/machine-learning/recommendation)
- [difflib Documentation](https://docs.python.org/3/library/difflib.html)

---

## 👨‍💻 Author
Machine Learning Practitioner | Recommendation Systems Enthusiast

**Note:** This project demonstrates the fundamentals of content-based recommendation systems, which form the backbone of many real-world recommendation engines used by Netflix, Amazon, and YouTube.

---

## 🎬 Try These Movies
Test the system with these popular movies:
- "The Dark Knight"
- "Inception"
- "The Matrix"
- "Interstellar"
- "The Godfather"
- "Forrest Gump"
- "Pulp Fiction"

Each will give you unique recommendations based on their distinct characteristics!
