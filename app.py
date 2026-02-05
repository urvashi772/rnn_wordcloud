# ======================================
# STREAMLIT APP
# PDF + IMAGE UPLOAD → RNN WORD CLOUD
# NO TensorFlow | NO PyTorch
# ======================================

import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import os

nltk.download("stopwords")

st.set_page_config(page_title="PDF RNN WordCloud", layout="centered")
st.title("📄 PDF → RNN Word Cloud Generator")
st.write("Upload a PDF and a PNG/JPG mask image")

# -------------------------------
# FUNCTIONS
# -------------------------------

def load_pdf_text_from_file(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    return [w for w in words if w not in stop_words and len(w) > 2]


@st.cache_resource
def load_glove(glove_path):
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def rnn_word_importance(words, embeddings):
    h_prev = np.zeros(100)
    importance = {}

    for word in words:
        if word not in embeddings:
            continue
        x = embeddings[word]
        h = np.tanh(x + 0.5 * h_prev)
        score = np.sum(np.abs(h))
        importance[word] = importance.get(word, 0) + score
        h_prev = h

    return importance


def generate_wordcloud(freq_dict, mask_image, output_name):
    img = Image.open(mask_image).convert("L")
    mask = np.array(img)

    # IMPORTANT: invert mask
    mask = np.where(mask > 200, 0, 255)

    wc = WordCloud(
        background_color="white",
        mask=mask,
        colormap="viridis",
        max_words=200,
        contour_width=2,
        contour_color="black"
    )

    wc.generate_from_frequencies(freq_dict)
    wc.to_file(output_name)

    return wc


# -------------------------------
# FILE UPLOAD UI
# -------------------------------

uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
uploaded_mask = st.file_uploader("🖼️ Upload Mask Image (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded_pdf and uploaded_mask:
    if st.button("🚀 Generate Word Cloud"):
        with st.spinner("Processing..."):

            # Load GloVe
            glove = load_glove("glove.6B.100d.txt")

            # PDF → Text
            text = load_pdf_text_from_file(uploaded_pdf)
            words = preprocess_text(text)

            # RNN Importance
            importance = rnn_word_importance(words, glove)

            # Save paths
            output_image = "wordcloud_output.png"

            # Generate cloud
            wc = generate_wordcloud(importance, uploaded_mask, output_image)

            st.success("✅ Word Cloud Generated & Saved")

            # Show image
            st.image(output_image, caption="Generated Word Cloud", use_column_width=True)

            # Download button
            with open(output_image, "rb") as f:
                st.download_button(
                    label="⬇️ Download Word Cloud",
                    data=f,
                    file_name="wordcloud_output.png",
                    mime="image/png"
                )

else:
    st.info("⬆️ Please upload both PDF and Mask Image")
