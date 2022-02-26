import pathlib
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sommelier Says", page_icon="🍷")

STREAMLIT_STATIC_PATH = (pathlib.Path(st.__path__[0]) / "static")

# Header and body text
st.title("Sommelier Says 🍷")
st.subheader("Just-for-fun app that predicts the points awarded to a wine,\
    as well as its price, based on its review.")
st.write("If you found this interesting, connect with me, Elizabeth Oda, via\
    [GitHub](https://github.com/elizabeth-oda) or\
    [LinkedIn.](https://www.linkedin.com/in/elizabethoda/)")


# Displays example reviews from the test set
st.subheader("Example reviews that you can copy and paste")
st.write(
    "You can **copy and paste any of the reviews below**, or upload your own.\
    Please note that the reviews shown here were not used to train the model.")

# Allows users to insert text
st.header("Try it out!")
review = st.text_area("Paste a wine review here")

# The model makes predictions and displays them
if review:
    img = load_img(png)
    result = predict(img)
    predictions = process_predict(result)
    top_three = dict(sorted(predictions.items(), key=lambda x: -x[1])[:3])
    st.header("Your Results")
    for l, p in top_three.items():
        st.subheader(l)
        st.write("Probability: " + str(round(p * 100, 1)) + "%")
