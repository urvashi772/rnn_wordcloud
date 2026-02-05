**📄 PDF to Masked Word Cloud Generator (RNN-based)**

This project generates custom-shaped word clouds from PDF documents using a pre-trained NLP model and an RNN-style sequential importance algorithm, without TensorFlow or PyTorch.

Users can upload:

📄 A PDF file

🖼️ A mask image (PNG / JPG)

And get a semantic, shape-based word cloud, downloadable as an image.

**🔥 Key Features**

✅ Upload PDF documents

✅ Upload PNG / JPG mask images

✅ Uses pre-trained GloVe embeddings

✅ Implements RNN-style word importance using NumPy

✅ Generates custom-shaped word clouds

✅ Saves output image locally

✅ Interactive Streamlit web app

❌ No TensorFlow

❌ No PyTorch

**🧠 How It Works (Pipeline)**
PDF → Text Extraction → Cleaning
    → Pre-trained GloVe Embeddings
    → RNN-style Sequential Scoring
    → Masked Word Cloud → Image Output

**🛠️ Tech Stack**

Python

NumPy

NLTK

PyPDF2

WordCloud

Matplotlib

Pillow

Streamlit

Pre-trained GloVe (Stanford NLP)

**📁 Project Structure**
project/
│── app.py
│── requirements.txt
│── glove.6B.100d.txt
│── README.md


📌 PDFs and mask images are uploaded directly through the Streamlit UI.

**📦 Installation**

1️⃣ Create Virtual Environment (Recommended)
python -m venv rnn_env
rnn_env\Scripts\activate   # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download Pre-trained GloVe Model

Download from:

https://nlp.stanford.edu/projects/glove/


Use:

glove.6B.100d.txt


Place it in the project folder.

▶️ Run the Application
streamlit run app.py


Open browser at:

http://localhost:8501

**🖼️ Mask Image Guidelines**

✔ Supported formats: PNG, JPG

✔ White area → words appear

✔ Black area → empty

✔ High contrast images work best

✔ Simple shapes give better results

Examples:

Heart ❤️

Brain 🧠

Cloud ☁️

India Map 🇮🇳

Logo shapes

**📤 Output**

Word cloud image is:

Displayed in browser

Saved locally (wordcloud_output.png)

Available for download

**💼 Resume / Interview Description**

“Developed an interactive Streamlit application that generates custom-shaped word clouds from PDF documents using pre-trained GloVe embeddings and an RNN-style NumPy model, without using TensorFlow or PyTorch.”

**🚀 Future Enhancements**

Multiple mask selection

Color theme selector

Transparent background export

Keyword frequency CSV export

Streamlit Cloud deployment

Multilingual PDF support
