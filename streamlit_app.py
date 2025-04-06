import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import os

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

# Specify a cache directory
cache_dir = "/tmp/model_cache"
os.makedirs(cache_dir, exist_ok=True)

if uploaded_file and direction:
    src_lang, tgt_lang = model_map[direction]
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    try:
        # Use the cache directory when loading the model and tokenizer
        # and force download to handle potential corrupted cache issues
        tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir, force_download=True)
        model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir, force_download=True)

        text = uploaded_file.read().decode("utf-8")
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.success("Translation Done!")
        st.download_button("Download Translation", translated_text, file_name="translated.txt")
    except Exception as e:
        st.error(f"An error occurred while loading the model or during translation: {e}")
