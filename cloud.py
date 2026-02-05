# ===============================
# PDF → Pretrained RNN → WordCloud
# MASK IMAGE (PNG / JPG)
# SAVE WORDCLOUD IMAGE
# NO TensorFlow | NO PyTorch
# ===============================

import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import os

nltk.download('stopwords')

# -------------------------------
# 1. LOAD PDF TEXT
# -------------------------------
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text


# -------------------------------
# 2. TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return words


# -------------------------------
# 3. LOAD GLOVE MODEL
# -------------------------------
def load_glove(glove_path):
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


# -------------------------------
# 4. RNN-STYLE IMPORTANCE
# -------------------------------
def rnn_word_importance(words, embeddings):
    hidden_size = 100
    h_prev = np.zeros(hidden_size)

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


# -------------------------------
# 5. GENERATE + SAVE WORD CLOUD
# -------------------------------
def generate_wordcloud(freq_dict, mask_path, output_name="wordcloud.png"):

    img = Image.open(mask_path).convert("L")
    mask = np.array(img)

    # IMPORTANT FIX (invert mask)
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

    # SAVE IMAGE
    wc.to_file(output_name)
    print(f"[SUCCESS] WordCloud saved as {output_name}")

    # SHOW IMAGE
    plt.figure(figsize=(8, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# -------------------------------
# 6. MAIN
# -------------------------------
def main():
    PDF_PATH = "Narendra_Modi.pdf"
    GLOVE_PATH = "glove.6B.100d.txt"
    MASK_PATH = "md.jpg"          # md.png OR md.jpg
    OUTPUT_IMAGE = "cloud_output.png"

    print("[INFO] Loading PDF...")
    text = load_pdf_text(PDF_PATH)

    print("[INFO] Preprocessing text...")
    words = preprocess_text(text)

    print("[INFO] Loading GloVe...")
    glove = load_glove(GLOVE_PATH)

    print("[INFO] Computing word importance...")
    importance = rnn_word_importance(words, glove)

    print("[INFO] Generating & saving word cloud...")
    generate_wordcloud(importance, MASK_PATH, OUTPUT_IMAGE)


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()
