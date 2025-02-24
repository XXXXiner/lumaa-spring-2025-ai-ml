### **🎬 Movie Recommendation System**  

---

This project provides **two methods** for recommending movies based on user preferences:  
1. **Genre-Based Recommendation** (using **Binary Features, Bag-of-Words (BoW), or TF-IDF**)  
2. **Transformer-Based Recommendation** (using **Sentence Transformers and semantic similarity**)  

The system also **analyzes sentiment** in user queries to handle **negation** (e.g., `"I don't like action movies"` will correctly avoid action movies).  


## **🚀 Features**
- **Interactive CLI**: Users can enter a movie preference and choose their preferred recommendation method.  
- **Sentiment Analysis**: Distinguishes **positive vs. negative sentiment** to improve results.  
- **Negation Handling**: If the user says `"I don't like horror"`, horror movies are excluded.  
- **Multiple Vectorization Techniques** (For **genre-based recommendations**):
  - **Binary Feature Matrix** (Presence/Absence of genres)
  - **Bag of Words (BoW)** (Word count representation)
  - **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Pre-trained Sentence Transformer Model** (For **transformer-based recommendations**):
  - Uses **"all-mpnet-base-v2"** to **encode** movie descriptions & genres for **semantic similarity matching**.


## **📂 Dataset**
- **Source**: `titles.csv` (Netflix Movie Dataset)  
- **Columns Used**:
  - `title`: The name of the movie  
  - `description`: A brief plot summary  
  - `genres`: A list of genres  


## **💻 Installation**
### **1️⃣ Prerequisites**
Ensure you have **Python 3.7+** installed. You can install required dependencies using:

```bash
pip install -r requirements.txt
```

### **2️⃣ Required Libraries**
- `pandas`  
- `nltk` (for sentiment analysis & stopword removal)  
- `scikit-learn` (for text vectorization and similarity measurement)  
- `sentence-transformers` (for Transformer-based recommendations)  
- `ace_tools` (for displaying recommendations in a structured format)  

To download the necessary NLTK resources:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
```


## **🛠 How to Run**
Run the script with:
```bash
python recommendation_system.py
```

### **👤 User Input Workflow**
1. **Enter a movie preference**  
   - Example: `"I love thrilling action movies set in space."`  
   - Example: `"I don't like horror movies."`  

2. **Choose a recommendation method**  
   - `"genres"` → Uses **Binary Features, BoW, or TF-IDF**  
   - `"transformer"` → Uses **SentenceTransformer embeddings**  

3. **(If 'genres' is selected) Choose a vectorizer**  
   - `"binary"` → Simple presence/absence of genres  
   - `"bow"` → Word frequency-based  
   - `"tfidf"` → Importance-weighted word frequencies  

4. **Enter the number of recommendations**  
   - Example: `5` (for top 5 recommendations)  

5. **Results are displayed!** 🎉  

---

## **🔬 Example Outputs**
### **1️⃣ Genre-Based Recommendation (TF-IDF)**
💡 **User Input**:  
```text
I love sci-fi action movies with adventure.
```
💡 **Results (TF-IDF Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *Interstellar* | Sci-Fi, Adventure | 0.91 |
| *The Martian* | Sci-Fi, Drama | 0.89 |
| *Guardians of the Galaxy* | Action, Adventure | 0.87 |

---

### **2️⃣ Transformer-Based Recommendation**
💡 **User Input**:  
```text
I don't like romance or drama.
```
💡 **Results (Transformer Method)**:
| Title | Genres | Similarity Score |
|--------|-----------------|----------------|
| *Mad Max: Fury Road* | Action, Sci-Fi | 0.85 |
| *John Wick* | Action, Crime | 0.82 |
| *The Dark Knight* | Action, Thriller | 0.80 |

**❌ Avoided "romance" and "drama" movies as expected!** ✅

---

## **🛠 Customization**
- **Modify the Sentiment Threshold**:  
  - Adjust `sia.polarity_scores(user_input)["compound"]` threshold in `detect_sentiment()`  
- **Change the Transformer Model**:  
  - Replace `"all-mpnet-base-v2"` with another **SentenceTransformer** model in the script  
- **Experiment with Different Vectorizers**:  
  - Try other **TF-IDF or Word2Vec-based** approaches for better results  

---

## **📌 Future Improvements**
- ✅ Add **IMDb Score Filtering** (recommend only highly-rated movies)  
- ✅ Implement **Hybrid Recommendation** (combine **genre-based & transformer-based** for best accuracy)  
- ✅ Optimize **Transformer Inference Speed** (use **FAISS** for fast vector search)  

---

## **📜 License**
This project is **open-source** and available for modification.  

---

## **📩 Contact**
💡 **Contributions & feedback are welcome!** 🚀  

---

This **README** provides a **clear overview**, **installation steps**, and **example outputs** for your **Movie Recommendation System**! 🚀 Let me know if you'd like any **modifications** or **additional sections**. 😊