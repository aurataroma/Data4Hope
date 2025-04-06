import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

st.set_page_config(
    page_title="Data4Hope Translator",
    page_icon="🌿",
    layout="centered"
)

model_map = {
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "Spanish to English": "Helsinki-NLP/opus-mt-es-en"  
}

st.title("Data4Hope English2Spanish Translator")
direction = st.selectbox("Choose translation direction", list(model_map.keys()))

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# Cache Dir
cache_dir = "/tmp/model_cache"
os.makedirs(cache_dir, exist_ok=True)

if uploaded_file and direction:
    model_name = model_map[direction]
    
    try:
        # Loading Modeling from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

        text = uploaded_file.read().decode("utf-8")
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.success("Translation Done!")
        st.download_button("Download Translation", translated_text, file_name="translated.txt")
    except Exception as e:
        st.error(f"An error occurred while loading the model or during translation: {e}")
