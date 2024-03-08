import pickle
from flask import Flask, render_template, request
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd

app = Flask(__name__)

# Define path to the CSV file
data_path = r'C:\Users\falcon\Desktop\paperr\papers.csv'

# Load example data from CSV (assuming it's a CSV file)
try:
    df = pd.read_csv(data_path)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    # Handle the error by skipping bad lines or taking appropriate action

# Create and fit CountVectorizer
cv = CountVectorizer(max_features=4000)
X_train_counts = cv.fit_transform(df['title'])  # Replace 'text_column' with your actual column name

# Create and fit TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Extract feature names after fitting CountVectorizer
feature_names = cv.get_feature_names_out()

with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

with open('tfidf_transformer.pkl', 'wb') as f:
    pickle.dump(tfidf_transformer, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(cv.get_feature_names_out(), f)



# Cleaning data:
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using",
                  "show", "result", "large",
                  "also", "one", "two", "three",
                  "four", "five", "seven", "eight", "nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    # Lower case
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    txt = [word for word in txt if word not in stop_words]
    # Remove words less than three letters
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatize
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']
    if document.filename == '':
        return render_template('index.html', error='No document selected')

    if document:
        text = document.read().decode('utf-8', errors='ignore')
        preprocessed_text = preprocess_text(text)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        return render_template('keywords.html', keywords=keywords)
    return render_template('index.html')

@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = []
        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords) == 20:  # Limit to 20 keywords
                    break
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
