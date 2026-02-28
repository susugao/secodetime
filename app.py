import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import random

# --- 1. 定義麥克風感測氣球 HTML/JS (用於負面情緒調節) ---
balloon_interactive_html = """
<div style="text-align: center; font-family: sans-serif;">
    <canvas id="balloonCanvas" width="300" height="250" style="border-radius: 15px; background-color: #f0f2f6;"></canvas>
    <div id="status" style="margin-top: 10px; font-weight: bold; color: #ff4b4b;">點擊下方按鈕開啟感應</div>
    <button id="startMic" style="padding: 10px 20px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer;">🎤 開始練習 (對著麥克風吹氣)</button>
</div>
<script>
    const canvas = document.getElementById('balloonCanvas');
    const ctx = canvas.getContext('2d');
    const statusText = document.getElementById('status');
    const startBtn = document.getElementById('startMic');
    let balloonRadius = 50;
    function draw(r) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, r, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b6b'; ctx.fill();
        ctx.beginPath(); ctx.moveTo(canvas.width/2, canvas.height/2 + r);
        ctx.lineTo(canvas.width/2, canvas.height/2 + r + 30); ctx.stroke();
    }
    draw(balloonRadius);
    startBtn.onclick = function() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
            statusText.innerText = "🌟 感應中！請開始吹氣..."; statusText.style.color = "#28a745"; startBtn.style.display = "none";
            let audioContext = new (window.AudioContext || window.webkitAudioContext)();
            let analyser = audioContext.createAnalyser();
            let microphone = audioContext.createMediaStreamSource(stream);
            let javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
            analyser.fftSize = 1024;
            microphone.connect(analyser); analyser.connect(javascriptNode); javascriptNode.connect(audioContext.destination);
            javascriptNode.onaudioprocess = function() {
                let array = new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(array);
                let values = 0; for (let i = 0; i < array.length; i++) { values += array[i]; }
                let average = values / array.length;
                if (average > 25) { if (balloonRadius < 110) balloonRadius += 2; } 
                else { if (balloonRadius > 50) balloonRadius -= 0.8; }
                draw(balloonRadius);
            };
        });
    };
</script>
"""

# --- 2. 資料紀錄函數 (包含姓名與班級) ---
def save_data(name, stu_class, user_input, emotion_label):
    file_path = "student_logs.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 建立包含五個欄位的資料表
    new_data = pd.DataFrame([[now, stu_class, name, user_input, emotion_label]], 
                            columns=["時間", "班級", "姓名", "學生輸入", "辨識情緒"])
    if os.path.exists(file_path):
        new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')

# --- 3. 網頁基礎設定與模型載入 ---
st.set_page_config(page_title="情緒小精靈-研究版", page_icon="🌈")
st.title("🌈 你的專屬情緒小精靈")

@st.cache_resource
def load_models():
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

feedback_dict = {
    "開心": {"msg": ["哇！你的好心情像陽光一樣閃亮！", "太好了，小精靈也想跟你一起跳舞！"], "action": "balloons"},
    "憤怒": {"msg": ["小精靈感覺到你心裡火辣辣的。", "呼～我們要不要一起把氣球吹大，把氣吐出來？"], "action": "warning"},
    "悲傷": {"msg": ["想哭也沒關係，小精靈會陪著你。", "抱一個！你今天辛苦了。"], "action": "info"},
    "平淡": {"msg": ["平穩的一天也是很珍貴的喔。", "心情靜靜的，感覺很舒服。"], "action": "snow"}
}

# --- 4. 學生基本資料輸入區 ---
st.markdown("### 📝 第一步：請填寫基本資料")
col1, col2 = st.columns(2)
with col1:
    student_class = st.text_input("你的班級：", placeholder="例如：三年一班")
with col2:
    student_name = st.text_input("你的名字：", placeholder="例如：小明")

st.divider()

# --- 5. 互動介面 (語音/文字) ---
st.markdown("### 🎤 第二步：對小精靈說說心情")
audio = mic_recorder(start_prompt="🎤 點我開始錄音", stop_prompt="🛑 說完了", key='recorder')
user_text = ""

if audio:
    with st.spinner('小精靈正在努力聽你說話...'):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio['bytes'])
        result = stt_model.transcribe("temp_audio.wav", fp16=False)
        user_text = result['text']
        st.success(f"小精靈聽到了：{user_text}")

manual_text = st.text_input("或是用打字的也可以：", value=user_text)
final_text = manual_text if manual_text else user_text

# --- 6. 核心邏輯：辨識 + 紀錄 + 回饋 ---
if final_text:
    # A. 辨識
    prediction = emo_classifier(final_text)[0]
    label = label_map.get(prediction['label'], "平淡")
    
    # B. 存檔 (將姓名與班級一起存入)
    save_data(student_name if student_name else "未填寫", 
              student_class if student_class else "未填寫", 
              final_text, label)
    
    # C. 視覺與對話回饋
    fb = feedback_dict.get(label, {"msg": ["謝謝你分享你的感覺。"], "action": None})
    st.divider()
    
    # 噴發特效
    if fb["action"] == "balloons": st.balloons()
    elif fb["action"] == "snow": st.snow()
    
    # 顯示結果
    st.subheader(f"小精靈覺得你現在：{label}")
    st.chat_message("assistant", avatar="🌈").write(random.choice(fb["msg"]))
    
    # D. 負面情緒介入練習
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.info("🌟 感覺有點不舒服嗎？小精靈陪你做個『氣球吹氣』練習")
        components.html(balloon_interactive_html, height=400)
        if st.button("我練習完了，感覺好多了！"):
            st.balloons()
            st.success("你真棒！成功找回平靜的能量囉！")

# --- 7. 老師管理區塊 (側邊欄) ---
with st.sidebar:
    st.title("📊 研究數據管理後台")
    st.caption("僅供廖長彥教授與研究者使用")
    if st.checkbox("顯示學生紀錄清單"):
        # 暫時加入這行來刪除舊格式檔案，跑通一次後就可以把這行刪掉
if os.path.exists("student_logs.csv"):
    os.remove("student_logs.csv")
        if os.path.exists("student_logs.csv"):
            df = pd.read_csv("student_logs.csv")
            st.dataframe(df)
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載完整資料報表 (CSV)", 
                data=csv_data, 
                file_name=f"情緒紀錄_{datetime.now().strftime('%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("目前尚無資料紀錄。")
