import streamlit as st
from transformers import pipeline

st.title("🌈 特教情緒小精靈測試版")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")

classifier = load_model()
text = st.text_input("請輸入一句心情：")

if text:
    res = classifier(text)[0]
    st.write(f"AI 辨識結果：{res['label']}")
