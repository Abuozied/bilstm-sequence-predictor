# Airline Tweet Sentiment (Streamlit + PyTorch Bi-LSTM)

## Structure
```
app.py                      # Streamlit app
requirements.txt
model/
  best_lstm_model.pth       # trained weights
  word_index.json           # vocabulary (word -> id)
  label_classes.json        # ["negative", "neutral", "positive"]
```

## Run
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the NLTK stopwords and wordnet data automatically.
