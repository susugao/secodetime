import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import random

# --- 1. 定義氣球動畫組件 ---
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
        ctx.lineTo(canvas.width/2, canvas.height/2 + r + 30); ctx.strokeStyle = '#555'; ctx.stroke();
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

# --- 2. 網頁設定與模型載入 ---
st.set_page_config(page_title="情緒小精靈-研究正式版", page_icon="🌈")

@st.cache_resource
def load_models():
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

# --- 3. 介面開始 ---
st.title("🌈 你的專屬情緒小精靈")

st.markdown("### 📝 第一步：請確認個人資料")
c1, c2, c3 = st.columns(3)
with c1:
    grade = st.selectbox("年級", [f"{i}年級" for i in range(1, 7)])
with c2:
    classroom = st.selectbox("班級", [f"{i}班" for i in range(1, 7)])
with c3:
    student_name = st.text_input("名字", placeholder="小明")

st.divider()

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

manual_text = st.text_input("或是你想用打字的也可以：", value=user_text)
final_text = manual_text if manual_text else user_text

# --- 4. 學生自評情緒 ---
st.markdown("### 🧐 第三步：你覺得你現在的心情是？")
student_self_label = st.radio(
    "請選擇最接近的心情：",
    ["開心", "平淡", "難過", "生氣", "害怕", "驚奇"],
    horizontal=True
)

# --- 5. 數據處理與回饋 ---
if st.button("🚀 點我送出給小精靈"):
    if final_text:
        prediction = emo_classifier(final_text)[0]
        ai_label = label_map.get(prediction['label'], "平淡")
        
        # 存檔邏輯
        file_path = "student_logs.csv"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 統一使用 7 個欄位
        new_data = pd.DataFrame([[now, grade, classroom, student_name, final_text, student_self_label, ai_label]], 
                                columns=["時間", "年級", "班級", "姓名", "內容", "自評情緒", "AI辨識情緒"])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        st.divider()
        st.subheader(f"分析結果：{ai_label}")
        
        if ai_label in ["憤怒", "悲傷", "厭惡"]:
            st.info("🌈 小精靈感覺到你不開心，我們來練習深呼吸吧！")
            components.html(balloon_interactive_html, height=400)
        else:
            st.balloons()
            st.success(f"謝謝 {student_name} 的分享！我已經記下來囉！")
    else:
        st.warning("請先輸入內容喔！")

# --- 6. 管理後台 (修正後的安全讀取版) ---
with st.sidebar:
    st.title("⚙️ 研究管理")
    if st.checkbox("開啟數據模式"):
        file_path = "student_logs.csv"
        if os.path.exists(file_path):
            try:
                # 嘗試讀取
                df = pd.read_csv(file_path)
                st.write("### 數據報表")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下載數據", csv, "data.csv", "text/csv")
            except Exception as e:
                st.error("⚠️ 偵測到舊格式資料衝突，系統已自動為您重置檔案。")
                os.remove(file_path) # 直接刪除有問題的舊檔案
                st.info("請重新整理網頁，新錄入的資料將會正常顯示。")
        else:
            st.info("目前尚無資料。")

    st.divider()
    # 隱藏的強制重置按鈕
    if st.button("🚨 強制重置資料庫"):
        if os.path.exists("student_logs.csv"):
            os.remove("student_logs.csv")
            st.rerun()
