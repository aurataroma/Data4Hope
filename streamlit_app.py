pip install streamlit transformers sentencepiece
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import tempfile

st.set_page_config(
    page_title="Data4Hope Translator",
    page_icon="🌿",
    layout="centered"
)

model_map = {
    "English to Spanish": ("en", "es"),
    "Spanish to English": ("es", "en")
}

st.title("Data4Hope English2Spanish Translator")
direction = st.selectbox("Choose translation direction", list(model_map.keys()))

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file and direction:
    src_lang, tgt_lang = model_map[direction]
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    text = uploaded_file.read().decode("utf-8")
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    st.success("Translation Done!")
    st.download_button("Download Translation", translated_text, file_name="translated.txt")

