import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import random

# --- 1. 優化後的呼吸氣球 (含虛線目標與穩定器) ---
balloon_logic_html = """
<div style="text-align: center; font-family: sans-serif; background: #ffffff; padding: 20px; border-radius: 20px;">
    <canvas id="balloonCanvas" width="300" height="350"></canvas>
    <div id="status" style="font-size: 20px; font-weight: bold; color: #ff4b4b; margin: 10px 0;">準備好練習呼吸了嗎？</div>
    <button id="startBtn" style="padding: 15px 30px; font-size: 18px; background: #ff4b4b; color: white; border: none; border-radius: 50px; cursor: pointer; box-shadow: 0 4px 15px rgba(255,75,75,0.3);">🎤 開始深呼吸練習</button>
</div>

<script>
    const canvas = document.getElementById('balloonCanvas');
    const ctx = canvas.getContext('2d');
    const statusText = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');

    let radius = 50;
    let targetRadius = 110;
    let mode = 'idle'; // idle, blowing (吹氣), inhaling (吸氣)
    let smoothedVolume = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 繪製目標虛線
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, targetRadius, 0, Math.PI * 2);
        ctx.setLineDash([5, 10]);
        ctx.strokeStyle = '#bbb';
        ctx.stroke();
        ctx.setLineDash([]);

        // 繪製氣球
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, radius, 0, Math.PI * 2);
        ctx.fillStyle = mode === 'inhaling' ? '#4dabf7' : '#ff6b6b';
        ctx.fill();
        
        // 氣球繩子
        ctx.beginPath();
        ctx.moveTo(canvas.width/2, canvas.height/2 + radius);
        ctx.lineTo(canvas.width/2, canvas.height/2 + radius + 40);
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 2;
        ctx.stroke();

        requestAnimationFrame(draw);
    }

    startBtn.onclick = async function() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        analyser.fftSize = 256;
        source.connect(analyser);

        startBtn.style.display = 'none';
        mode = 'blowing';
        statusText.innerText = "💨 吹氣：讓氣球碰到虛線！";

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        function process() {
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for(let i=0; i<dataArray.length; i++) sum += dataArray[i];
            let volume = sum / dataArray.length;

            // 平滑化處理：避免忽大忽小
            smoothedVolume = smoothedVolume * 0.8 + volume * 0.2;

            if (mode === 'blowing') {
                if (smoothedVolume > 20) {
                    radius += 0.8;
                } else {
                    radius -= 0.2;
                }
                
                if (radius >= targetRadius) {
                    mode = 'inhaling';
                    statusText.innerText = "🌈 成功！現在慢慢吸氣...";
                    statusText.style.color = "#4dabf7";
                }
            } else if (mode === 'inhaling') {
                // 吸氣模式下氣球自動緩慢縮小
                radius -= 0.5;
                if (radius <= 50) {
                    mode = 'blowing';
                    statusText.innerText = "💨 再吹一次氣！";
                    statusText.style.color = "#ff4b4b";
                }
            }
            
            radius = Math.max(50, Math.min(radius, 130));
            requestAnimationFrame(process);
        }
        process();
    };
    draw();
</script>
"""

# --- 2. 核心功能 ---
st.set_page_config(page_title="情緒小精靈-研究版", page_icon="🌈")

@st.cache_resource
def load_models():
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

# --- 3. 介面設計 ---
st.title("🌈 你的專屬情緒小精靈")

# 學生資料
st.markdown("### 📝 第一步：請確認個人資料")
c1, c2, c3 = st.columns(3)
with c1:
    grade = st.selectbox("年級", [f"{i}年級" for i in range(1, 7)])
with c2:
    classroom = st.selectbox("班級", [f"{i}班" for i in range(1, 7)])
with c3:
    student_name = st.text_input("名字", placeholder="小明")

st.divider()

# 錄音區
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

# 自評
st.markdown("### 🧐 第三步：你覺得你現在的心情是？")
student_self_label = st.radio("選擇心情：", ["開心", "平淡", "難過", "生氣", "害怕", "驚奇"], horizontal=True)

# 提交
if st.button("🚀 點我送出給小精靈"):
    if final_text:
        prediction = emo_classifier(final_text)[0]
        ai_label = label_map.get(prediction['label'], "平淡")
        
        # 存檔
        file_path = "student_logs.csv"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[now, grade, classroom, student_name, final_text, student_self_label, ai_label]], 
                                columns=["時間", "年級", "班級", "姓名", "內容", "自評情緒", "AI辨識情緒"])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        # 顯示回饋
        st.session_state['last_label'] = ai_label
        st.session_state['submitted'] = True
        
        # 強制重置錄音組件
        st.rerun()

# --- 4. 動態顯示回饋區 (放在主迴圈避免不見) ---
if st.session_state.get('submitted'):
    label = st.session_state.get('last_label', '平淡')
    st.divider()
    st.subheader(f"分析結果：{label}")
    
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.info("🌟 氣球練習時間：吹氣讓氣球碰到虛線，然後跟著它慢慢吸氣。")
        components.html(balloon_logic_html, height=450)
    else:
        st.balloons()
        st.success("紀錄成功！你今天做得很棒！")
    
    if st.button("完成練習"):
        st.session_state['submitted'] = False
        st.rerun()

# --- 5. 研究後台 ---
with st.sidebar:
    st.title("⚙️ 研究管理")
    if st.checkbox("開啟數據模式"):
        if os.path.exists("student_logs.csv"):
            try:
                df = pd.read_csv("student_logs.csv")
                st.dataframe(df)
                st.download_button("📥 下載數據", df.to_csv(index=False).encode('utf-8-sig'), "data.csv")
            except:
                st.error("格式衝突，請重置。")
    if st.button("🚨 強制重置資料庫"):
        if os.path.exists("student_logs.csv"): os.remove("student_logs.csv")
        st.rerun()
