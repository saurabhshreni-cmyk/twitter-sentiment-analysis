# streamlit_app.py
import os
import re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------- Config -----------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECT_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CSV_PATH = Path("Tweets.csv")  # put Tweets.csv here if you want app to train automatically

# ----------------- Preprocessing (safe, no NLTK required) -----------------
STOP_WORDS = {
    "the","and","a","an","is","it","to","for","of","in","on","this","that","i","you",
    "we","they","me","my","so","but","not","have","has","be","was","were"
}

def clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_process(text: str) -> str:
    ct = clean_text_basic(text)
    toks = [t for t in ct.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(toks)

# ----------------- Model helpers -----------------
def train_quick_model_from_csv(csv_path: Path, model_path: Path, vect_path: Path):
    st.info("Training model from CSV. This will take ~30s or less for the airline dataset.")
    df = pd.read_csv(csv_path, low_memory=True)
    # normalize names
    if 'text' not in df.columns:
        for c in df.columns:
            if 'tweet' in c.lower() or 'text' in c.lower():
                df = df.rename(columns={c: 'text'})
                break
    if 'airline_sentiment' in df.columns:
        df = df.rename(columns={'airline_sentiment': 'label'})
    elif 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})

    if 'text' not in df.columns or 'label' not in df.columns:
        raise RuntimeError("CSV must contain 'text' and a label column like 'airline_sentiment' or 'sentiment'.")

    df = df[['text','label']].dropna().astype(str)
    # preprocess
    df['clean_text'] = df['text'].apply(tokenize_and_process)

    X = df['clean_text']
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vect = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X_train_vec = vect.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    joblib.dump(clf, model_path)
    joblib.dump(vect, vect_path)
    return clf, vect

def load_or_train_model():
    if MODEL_PATH.exists() and VECT_PATH.exists():
        clf = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
        return clf, vect, "loaded"
    else:
        if CSV_PATH.exists():
            clf, vect = train_quick_model_from_csv(CSV_PATH, MODEL_PATH, VECT_PATH)
            return clf, vect, "trained"
        else:
            return None, None, "missing"

# ----------------- Ensure session state keys -----------------
if 'tweet_input' not in st.session_state:
    st.session_state['tweet_input'] = ""
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'probability' not in st.session_state:
    st.session_state['probability'] = None
if 'status' not in st.session_state:
    st.session_state['status'] = None

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("Twitter Sentiment Analyzer")
st.write("Paste a tweet and get sentiment prediction (TF-IDF + LogisticRegression).")

# Load or train model
with st.spinner("Loading model..."):
    try:
        clf, vect, status = load_or_train_model()
    except Exception as e:
        clf, vect, status = None, None, "error"
        st.error(f"Model load/train error: {e}")
        st.stop()

st.session_state['status'] = status
if status == "trained":
    st.success("Model trained from Tweets.csv and saved to ./models/")
elif status == "loaded":
    st.success("Model loaded from ./models/")

# Sidebar
with st.sidebar:
    st.header("Model")
    st.write(f"Status: **{st.session_state['status']}**")
    if clf is not None:
        st.write("Model type:", type(clf).__name__)
    st.markdown("---")
    st.header("Controls")
    if st.button("Retrain from CSV"):
        if CSV_PATH.exists():
            clf, vect = train_quick_model_from_csv(CSV_PATH, MODEL_PATH, VECT_PATH)
            st.success("Retrained and saved model.")
        else:
            st.error("Tweets.csv not found in project root. Place the CSV and try again.")
    st.markdown("---")
    st.write("Tips:")
    st.write("- If app cannot find models, add `Tweets.csv` (airline dataset) to the folder and press Retrain.")
    st.write("- Saved models are in `./models/`.")

# Main area: input and examples
col_left, col_right = st.columns([2,1])

with col_left:
    # show text_area bound to session state
    tweet = st.text_area(
        "Enter tweet text",
        value=st.session_state['tweet_input'],
        placeholder="Type or paste a tweet here...",
        height=160,
        key="tweet_text_area"
    )

    # Analyze button (explicit)
    if st.button("Analyze"):
        text_to_analyze = tweet.strip()
        if not text_to_analyze:
            st.error("Please enter a tweet.")
        else:
            st.session_state['tweet_input'] = text_to_analyze
            cleaned = tokenize_and_process(text_to_analyze)
            vec = vect.transform([cleaned])
            pred = clf.predict(vec)[0]
            prob = None
            if hasattr(clf, "predict_proba"):
                prob = float(clf.predict_proba(vec).max())
            st.session_state['prediction'] = pred
            st.session_state['probability'] = prob
            # show output immediately
            if pred == "positive":
                st.success(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
            elif pred == "negative":
                st.error(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
            else:
                st.info(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
            st.write("**Preprocessed text:**", cleaned)

with col_right:
    st.subheader("Quick examples")
    # Example buttons set session_state and immediately perform the prediction
    if st.button("Example: positive"):
        example = "I love the new in-flight entertainment and seats!"
        st.session_state['tweet_input'] = example
        cleaned = tokenize_and_process(example)
        vec = vect.transform([cleaned])
        pred = clf.predict(vec)[0]
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = float(clf.predict_proba(vec).max())
        st.session_state['prediction'] = pred
        st.session_state['probability'] = prob
        if pred == "positive":
            st.success(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        elif pred == "negative":
            st.error(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        else:
            st.info(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        st.write("**Preprocessed text:**", cleaned)

    if st.button("Example: negative"):
        example = "Terrible service, lost my luggage and rude staff."
        st.session_state['tweet_input'] = example
        cleaned = tokenize_and_process(example)
        vec = vect.transform([cleaned])
        pred = clf.predict(vec)[0]
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = float(clf.predict_proba(vec).max())
        st.session_state['prediction'] = pred
        st.session_state['probability'] = prob
        if pred == "positive":
            st.success(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        elif pred == "negative":
            st.error(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        else:
            st.info(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        st.write("**Preprocessed text:**", cleaned)

    if st.button("Example: neutral"):
        example = "Flight departed on time, boarding was normal."
        st.session_state['tweet_input'] = example
        cleaned = tokenize_and_process(example)
        vec = vect.transform([cleaned])
        pred = clf.predict(vec)[0]
        prob = None
        if hasattr(clf, "predict_proba"):
            prob = float(clf.predict_proba(vec).max())
        st.session_state['prediction'] = pred
        st.session_state['probability'] = prob
        if pred == "positive":
            st.success(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        elif pred == "negative":
            st.error(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        else:
            st.info(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
        st.write("**Preprocessed text:**", cleaned)

# Show last prediction summary if present
if st.session_state['prediction'] is not None and st.session_state['tweet_input']:
    st.markdown("---")
    pred = st.session_state['prediction']
    prob = st.session_state['probability']
    st.write("### Last prediction")
    if pred == "positive":
        st.success(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
    elif pred == "negative":
        st.error(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
    else:
        st.info(f"Prediction: {pred.upper()} {'(%.1f%%)' % (prob*100) if prob else ''}")
    st.write("**Text:**", st.session_state['tweet_input'])

st.markdown("---")
st.write("Model info: TF-IDF (max 3000 features) + LogisticRegression. Preprocessing: lowercasing, remove links/mentions, basic tokenization + stopword removal.")
