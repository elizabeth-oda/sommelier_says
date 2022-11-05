import streamlit as st
import numpy as np
import pandas as pd
from sommelier_says.data_processing import PreprocessStrings
import nltk
from nltk import word_tokenize
import pathlib
from os.path import join, isfile, dirname
from os import listdir
from PIL import Image
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

# Downloads essential NLTK libraries
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(page_title="Sommelier Says", page_icon="🍷")

STREAMLIT_STATIC_PATH = (pathlib.Path(st.__path__[0]) / "static")

# Header and body text
st.title("Sommelier Says 🍷")
st.subheader(
    "Just-for-fun app that predicts the points awarded to a wine\
    based on its review."
)
st.write(
    "**The details:** Wines are awarded points on a\
    [scale of 1-100](https://www.winespectator.com/articles/scoring-scale)\
    based on their quality. This app uses Word2Vec and LSTM to predict\
    the points awarded to a wine based on only its review. The model was\
    trained on more than ninety-thousand reviews from wine critics. The\
    mean square error is less than 3 points."
)
st.write(
    "If you found this interesting, connect with me, Elizabeth Oda, via\
    [GitHub](https://github.com/elizabeth-oda) or\
    [LinkedIn.](https://www.linkedin.com/in/elizabethoda/)"
)

# Insert text for reviews
st.header("Try it out!")
review = st.text_area("Paste a wine review here")

file_dir = dirname(__file__)

# The model makes predictions and displays them
if len(review) > 0:
    rev = PreprocessStrings.lower(review)
    rev = PreprocessStrings.punct(rev)
    rev = PreprocessStrings.remove_wine_stopwords(rev)
    wv_model = Word2Vec.load(join(file_dir, "models/wv_model200.model"))
    rev_pad = PreprocessStrings.embed_review(wv_model, rev)
    lstm_model = load_model(join(file_dir, "models/model4_points.h5"))
    result = lstm_model.predict(rev_pad)
    points = str(round(result[0][0]))
    st.subheader("Sommelier says... " + points + " points!")

# Displays example reviews from the test set
st.subheader("Example reviews that you can copy and paste")
st.write(
    "You can **copy and paste any of the reviews below**, or upload your own.\
    Please note that these reviews were not used to train the model."
)

X_test = pd.read_csv('data/X_test_wine.csv')
y_test = pd.read_csv('data/y_test_wine.csv')

if st.button("Load example reviews"):
    rand_five = np.random.randint(low=0, high=len(X_test['description']), size=5)
    for i in range(len(rand_five)):
        st.write(X_test['description'][rand_five[i]])
        st.write("Points: ", y_test['points'][rand_five[i]].astype('str'))
