import streamlit as st
from transformers import pipeline

# 1. 網頁基礎設定
st.set_page_config(page_title="情緒小精靈", page_icon="🌈")
st.title("🌈 你的專屬情緒小精靈")

# 2. 定義標籤翻譯字典 (修正你看到的 LABEL_0 問題)
# 根據 Johnson8187 模型的設定進行對應
label_map = {
    "LABEL_0": "平淡",
    "LABEL_1": "關切",
    "LABEL_2": "開心",
    "LABEL_3": "憤怒",
    "LABEL_4": "悲傷",
    "LABEL_5": "疑問",
    "LABEL_6": "驚奇",
    "LABEL_7": "厭惡"
}

# 情緒對應的圖示與建議
emotion_advice = {
    "開心": {"emoji": "🥳", "color": "green", "text": "太棒了！分享這份喜悅給老師吧！"},
    "憤怒": {"emoji": "😤", "color": "red", "text": "小精靈感覺到你在生氣，我們一起深呼吸三次好嗎？"},
    "悲傷": {"emoji": "🥺", "color": "blue", "text": "想哭也沒關係，小精靈會在這裡陪著你。"},
    "平淡": {"emoji": "😐", "color": "gray", "text": "平平穩穩的一天也很好喔！"},
    "驚奇": {"emoji": "😮", "color": "orange", "text": "哇！發生了什麼意想不到的事嗎？"}
}

# 3. 載入 AI 模型
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")

classifier = load_model()

# 4. 互動介面
st.write("### 嘿！今天在學校過得好嗎？")
text = st.text_input("在這裡寫下（或說出）你的心情：", placeholder="例如：今天體育課很好玩...")

if text:
    # AI 辨識
    prediction = classifier(text)[0]
    raw_label = prediction['label']
    conf_score = prediction['score']
    
    # 轉換成中文名稱
    chinese_label = label_map.get(raw_label, "神秘情緒")
    
    # 取得顯示資訊
    info = emotion_advice.get(chinese_label, {"emoji": "🤖", "color": "black", "text": "小精靈正在努力理解你的感覺..."})
    
    # 5. 視覺化呈現
    st.divider()
    
    # 顯示大大的圖示
    st.markdown(f"<h1 style='font-size: 100px; text-align: center;'>{info['emoji']}</h1>", unsafe_allow_html=True)
    
    # 顯示辨識結果
    st.subheader(f"小精靈覺得你現在：**{chinese_label}**")
    st.info(info['text'])
    
    # 6. 加入簡單的語音合成 (TTS) - 廖老師會喜歡這個教育輔助功能
    # 利用 HTML5 的語音 API，不佔伺服器資源
    tts_script = f"""
    <script>
        var msg = new SpeechSynthesisUtterance();
        msg.text = "小精靈覺得你現在感覺{chinese_label}。{info['text']}";
        msg.lang = 'zh-TW';
        window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(tts_script, height=0)

st.divider()
st.caption("本系統為中央大學網學所研究原型，僅供教學實踐使用。")
