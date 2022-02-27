import streamlit as st
import numpy as np
from wine_prices.data import Review
import nltk
import pathlib
from os.path import join, isfile, dirname
from os import listdir
from PIL import Image
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sommelier Says", page_icon="🍷")

STREAMLIT_STATIC_PATH = (pathlib.Path(st.__path__[0]) / "static")

# Header and body text
st.title("Sommelier Says 🍷")
st.subheader(
    "Just-for-fun app that predicts the points awarded to a wine\
    based on its review."
)
st.write(
    "If you found this interesting, connect with me, Elizabeth Oda, via\
    [GitHub](https://github.com/elizabeth-oda) or\
    [LinkedIn.](https://www.linkedin.com/in/elizabethoda/)"
)

# Insert text for reviews
st.header("Try it out!")
review = st.text_area("Paste a wine review here")

# The model makes predictions and displays them
if len(review) > 0:
    rev = Review.lower(review)
    rev = Review.punct(rev)
    rev = Review.remove_wine_stopwords(rev)
    wv_model = Word2Vec.load(join(dirname(__file__), "wv_model.model"))
    rev_pad = Review.embed_review(wv_model, rev)
    lstm_model = load_model(join(dirname(__file__), "model3_points.h5"))
    result = lstm_model.predict(rev_pad)
    st.write(result)

# Finds filepath for photos
file_dir = join(dirname(__file__), "photos")

# Displays example reviews from the test set
st.subheader("Example reviews that you can copy and paste")
st.write(
    "You can **copy and paste any of the reviews below**, or upload your own.\
    Please note that these reviews were not used to train the model."
)

# Review-photo set 1
col1, col2 = st.columns([1,2])
col1.image(join(file_dir, "jeff-siepman-unsplash.jpg"), width=180)
col2.write(
    "**Example 1:**\
    Clove, exotic spice, dark chocolate and cured meat come to the forefront\
    of this rich, plush Barbera d'Alba. There's power and bright acidity here\
    and the wine tastes rich, smooth and enduring on the palate. Pair it with\
    cheese-stuffed ravioli."
)
col2.subheader("Points: 91")

# Review-photo set 2
col3, col4 = st.columns([1,2])
col3.write("")
col3.write(
    "**Example 2:**\
    French oak, menthol and plum aromas waft out of the glass. The taut,\
    rather light-bodied palate offers tart red cherry, star anise and coffee\
    bean alongside grainy tannins that leave a grippy, astringent finish."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )
col3.subheader("Points: 88")
col3.write(
    "**Example 3:**\
    This is such a refreshing take on the variety, still robust and rich in\
    body, yet possessed of firm acidity and a crispness of cherry kirsch\
    and pomegranate. It practically shouts violets and roses on the nose,\
    following through on the palate with a delicacy of power that's entirely\
    surprising. The finish is a mix of allspice and white pepper."                                                                                                                                                                                                                                                                                                                                                                                                                                                                              )
col3.subheader("Points: 92")
col4.image(join(file_dir, "klara-kulikova-unsplash.jpg"))

# Review-photo set 3
col5, col6 = st.columns([1,2])
col5.image(join(file_dir, "big-dodzy-unsplash.jpg"))
col6.write("")
col6.write(
    "**Example 4:**\
    Whiffs of sweet curry spice and fennel are enticing on the nose of this\
    earthy, rather savory Riesling from Rock Stream. Lean in profile and\
    vibrantly acidic with a squeaky, lemony sheen, it's the perfect pairing\
    for subtly spiced South Indian cuisine."                                                                                                                                                                                                                                                                        )
col6.subheader("Points: 85")
