# 🎬 Movie Recommendation System

This project provides **two methods** for recommending movies based on user preferences:

1. **Genre-Based Recommendation** (using **Binary Features, Bag-of-Words (BoW), or TF-IDF**)
2. **Transformer-Based Recommendation** (using **Sentence Transformers and semantic similarity**)

The system also **analyzes sentiment** in user queries to handle **negation** (e.g., `"I don't like action movies"` will correctly avoid action movies).  


## 🚀 Features
- **Interactive CLI**: Users can enter a movie preference and choose their preferred recommendation method.  
- **Sentiment Analysis**: Distinguishes **positive vs. negative sentiment** to improve results.  
- **Negation Handling**: If the user says `"I don't like horror"`, horror movies would be excluded.  
- **3 Vectorization Techniques** (For **genre-based recommendations**):
  - **Binary Feature Matrix** (Presence/Absence of genres)
  - **Bag of Words (BoW)** (Word count representation)
  - **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Pre-trained Sentence Transformer Model** (For **genre and description-based recommendations**):
  - Uses **"all-mpnet-base-v2"** to **encode** movie descriptions & genres for **semantic similarity matching**.


## 🔥 Key Advantages
🔹 **More Accurate Recommendations**:  
   - Unlike simple keyword matching, this system **understands context and semantics** to suggest movies that match the user's actual intent.  

🔹 **Handles Negation Correctly**:  
   - If a user says **"I don't like horror"**, the system **removes horror movies** instead of misinterpreting it as liking horror.  

🔹 **4 Powerful Recommendation Methods**:  
   - **Genre-Based (TF-IDF, BoW, Binary Features)** → Fast and interpretable.  
   - **Transformer-Based (Semantic Similarity)** → More advanced and context-aware.  

🔹 **Sentiment-Aware Filtering**:  
   - If the user input is **positive**, the system recommends movies **most similar** to their input.  
   - If the input is **negative**, the system suggests movies **least similar** to avoid unwanted genres.  

🔹 **User Control Over Vectorization & Methods**:  
   - The user can choose between **Binary Features, BoW, TF-IDF, or Transformer embeddings** for recommendations.  

🔹 **Handles Multi-Genre & Combined Descriptions**:  
   - The transformer model uses **movie descriptions + genres** for improved recommendation quality.  


## 📂 Dataset
- **Source**: `titles.csv`([Netflix Movie Dataset](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies/data))  
- **Columns Used**:
  - `title`: The name of the movie  
  - `description`: A brief plot summary  
  - `genres`: A list of genres  


## 💻 Installation
### 1️⃣ Prerequisites
Ensure you have **Python 3.11** installed. You can install required dependencies using:

```bash
pip install -r requirements.txt
```

### 2️⃣ Required Libraries
- `pandas`  
- `nltk` (for sentiment analysis & stopword removal)  
- `scikit-learn` (for text vectorization and similarity measurement)  
- `sentence-transformers` (for Transformer-based recommendations)  
- `ace_tools_open` (for displaying recommendations in a structured format)  


## 🛠 How to Run
Run the script with:
```bash
python recommendation.py
```

### 👤 User Input Workflow
1. **Enter a movie preference**  
   - Example: `"I love thrilling action movies set in space."`  
   - Example: `"I don't like horror movies."`  

2. **Choose a recommendation method**  
   - `"vectorization"` → Uses **Binary Features, BoW, or TF-IDF**  
   - `"transformer"` → Uses **SentenceTransformer embeddings**  

3. **(If 'vectorization' is selected) Choose a vectorizer**  
   - `"binary"` → Simple presence/absence of genres  
   - `"bow"` → Word frequency-based  
   - `"tfidf"` → Importance-weighted word frequencies  

4. **Enter the number of recommendations**  
   - Example: `5` (for top 5 recommendations)  

5. **Results are displayed!** 🎉  


## 🔬 Example Outputs
### 1️⃣ Vectorization-Based Recommendation Using Genre(TF-IDF)
🤩 **User Input (Positive Sentiment)**:  
```text
I love sci-fi action movies with adventure.
```
💡 **Results (TF-IDF Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *You vs. Wild: Out Cold* | action | 1.0 |
| *Rogue Warfare: The Hunt* | action | 1.0 |
| *Black* | action | 1.0 |

---

😣 **User Input (Negative Sentiment)**:  
```text
I don't like thrilling action movies set in space.
```
💡 **Results (TF-IDF Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *Taxi Driver* | drama, crime | 0.0 |
| *Roped* | drama, romance, family, comedy | 0.0 |
| *A Remarkable Tale* | comedy, drama, european | 0.0 |

**❌ Avoided "thrilling" and "action" movies as expected!** ✅

---

### 2️⃣ Transformer-Based Recommendation Using Genre and Description

🤩 **User Input (Positive Sentiment)**:  
```text
I love romantic and comedic movie! 
```
💡 **Results (TF-IDF Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *Ajab Prem Ki Ghazab Kahani* | comedy, action, romance | 0.6305 |
| *Disconnect* | romance, comedy, scifi | 0.6124 |
| *Barfi!* | drama romance comedy | 0.5884 |

---

😣 **User Input (Negative Sentiment)**:  
```text
I hate scary historical movies.
```
💡 **Results (Transformer Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *Untold: Breaking Point* | sport, documentation | -0.1802 |
| *Middle of Nowhere* | drama | -0.1109 |
| *Kiss the Ground* | documentation | -0.0994 |

**❌ Avoided "scary" and "historical" movies as expected!** ✅


## 🛠 Customization
- **Modify the Sentiment Threshold**:  
  - You can adjust `sia.polarity_scores(user_input)["compound"]` threshold in `detect_sentiment()`. The threshold now is set as 0.65
- **Change the Transformer Model**:  
  - You can replace `"all-mpnet-base-v2"` with another **SentenceTransformer** model in the script if you want
- **Experiment with Different Vectorizers**:  
  - You can try other **TF-IDF or Word2Vec-based** approaches for better results if you want


## 📌 Future Improvements
- ✅ Add **IMDb Score Filtering** (recommend only highly-rated movies)  
- ✅ Implement **Hybrid Recommendation** (combine **genre-based & transformer-based** for best accuracy)  
- ✅ Optimize **Transformer Inference Speed** (use **FAISS** for fast vector search)  


## 🎦 Demo 
This demo is [Here](https://youtu.be/6MaHbtubckw)
## 🥳 Salary Expectation
Hourly: $25~40/h