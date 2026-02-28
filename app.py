import streamlit as st
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import os

# 1. 網頁基礎設定
st.set_page_config(page_title="情緒小精靈-語音版", page_icon="🌈")
st.title("🌈 你的專屬情緒小精靈 (語音版)")

# 2. 載入模型 (增加 Whisper 語音辨識)
@st.cache_resource
def load_models():
    # 情緒辨識模型
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    # 語音轉文字模型 (選用 base 等級，兼顧速度與準確度)
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()

# 標籤翻譯字典
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

# 3. 語音輸入介面
st.write("### 點擊麥克風，對小精靈說說話吧！")
audio = mic_recorder(start_prompt="🎤 開始錄音", stop_prompt="🛑 說完了", key='recorder')

user_text = ""

# 當有錄音資料時進行處理
if audio:
    with st.spinner('小精靈正在努力聽你說話...'):
        # 將錄音資料存為暫存檔
        with open("temp_audio.wav", "wb") as f:
            f.write(audio['bytes'])
        
        # 使用 Whisper 轉文字
        result = stt_model.transcribe("temp_audio.wav", fp16=False)
        user_text = result['text']
        st.success(f"小精靈聽到了：{user_text}")

# 4. 保留文字輸入作為備案 (雙軌輸入)
manual_text = st.text_input("或是你想用打字的也可以：", value=user_text)
final_text = manual_text if manual_text else user_text

# 5. 情緒分析與回饋 (沿用之前的邏輯)
if final_text:
    prediction = emo_classifier(final_text)[0]
    chinese_label = label_map.get(prediction['label'], "未知")
    
    st.divider()
    st.markdown(f"<h1 style='text-align: center;'>🥳</h1>", unsafe_allow_html=True) # 這裡可依標籤換圖
    st.subheader(f"小精靈覺得你現在：{chinese_label}")
