import json
import re
import string
from pathlib import Path

import emoji
import nltk
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pack_padded_sequence

# ---------------- Settings (must match the training notebook) ----------------
MODEL_DIR = Path(__file__).parent / "model"
VOCAB_SIZE = 10000
MAX_LEN = 50
PAD, OOV = 0, 1
EMBED_DIM, HIDDEN = 128, 64

st.set_page_config(page_title="Airline Tweet Sentiment", page_icon="✈️")


# ---------------- Model definition (same as the notebook) ----------------
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_state=HIDDEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_state, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_state * 2, 3)

    def forward(self, x):
        lengths = (x != PAD).sum(dim=1).clamp(min=1).cpu()
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


# ---------------- Cached loaders ----------------
@st.cache_resource
def load_nlp():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource
def load_artifacts():
    model = SentimentLSTM()
    state = torch.load(MODEL_DIR / "best_lstm_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(MODEL_DIR / "word_index.json") as f:
        word_index = json.load(f)
    with open(MODEL_DIR / "label_classes.json") as f:
        classes = json.load(f)
    return model, word_index, classes


# ---------------- Preprocessing (same steps as the notebook) ----------------
def clean_text(text, stop_words, lemmatizer):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def encode(clean, word_index):
    ids = [word_index.get(w, OOV) for w in clean.split()][:MAX_LEN]
    ids += [PAD] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def predict(text):
    stop_words, lemmatizer = load_nlp()
    model, word_index, classes = load_artifacts()
    clean = clean_text(text, stop_words, lemmatizer)
    with torch.no_grad():
        probs = torch.softmax(model(encode(clean, word_index)), dim=1)[0].numpy()
    return clean, classes, probs


# ---------------- UI ----------------
st.title("✈️ Airline Tweet Sentiment")
st.write("Type an English tweet about an airline and the Bi-LSTM model will classify it "
         "as **negative**, **neutral** or **positive**.")

EXAMPLES = {
    "😡 Complaint": "My flight was delayed 4 hours and nobody told us anything. Worst service ever!",
    "😐 Neutral": "Does anyone know if the flight to Dallas leaves from terminal B?",
    "😊 Praise": "Thank you for the amazing crew today, the flight was smooth and on time!",
}

if "tweet" not in st.session_state:
    st.session_state.tweet = ""

cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.tweet = text

tweet = st.text_area("Tweet text", key="tweet", height=120,
                     placeholder="e.g. @united thanks for the great flight!")

if st.button("Analyze", type="primary"):
    if not tweet.strip():
        st.warning("Please enter some text first.")
    else:
        clean, classes, probs = predict(tweet)
        best = int(probs.argmax())
        icons = {"negative": "😡", "neutral": "😐", "positive": "😊"}
        st.subheader(f"{icons.get(classes[best], '')} {classes[best].capitalize()}  "
                     f"({probs[best] * 100:.1f}%)")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=classes))
        with st.expander("Cleaned text the model saw"):
            st.code(clean or "(empty after cleaning)")
        if not clean:
            st.info("Nothing was left after cleaning (only stop words/punctuation), "
                    "so this result is unreliable.")
