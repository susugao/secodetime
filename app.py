import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import random

# --- 1. 呼吸氣球 HTML (防誤觸穩定版) ---
balloon_logic_html = """
<div style="text-align: center; font-family: sans-serif; background: #ffffff; padding: 10px; border-radius: 20px;">
    <canvas id="balloonCanvas" width="300" height="400"></canvas>
    <div id="status" style="font-size: 20px; font-weight: bold; color: #ff4b4b; margin: 10px 0;">準備好練習呼吸了嗎?</div>
    <div id="debug" style="font-size: 12px; color: #ccc;">環境音量: 0 (需超過 40 才會長大)</div>
    <button id="startBtn" style="padding: 15px 30px; font-size: 18px; background: #ff4b4b; color: white; border: none; border-radius: 50px; cursor: pointer; box-shadow: 0 4px 15px rgba(255,75,75,0.3);">🎤 開始深呼吸練習</button>
</div>

<script>
    var canvas = document.getElementById('balloonCanvas');
    var ctx = canvas.getContext('2d');
    var radius = 50;
    var targetRadius = 110;
    var mode = 'idle';
    var smoothedVolume = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // 繪製目標圈
        ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, targetRadius, 0, Math.PI * 2);
        ctx.setLineDash([5, 8]); ctx.strokeStyle = '#dddddd'; ctx.stroke(); ctx.setLineDash([]);
        // 繪製氣球
        ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, radius, 0, Math.PI * 2);
        ctx.fillStyle = (mode === 'inhaling') ? '#4dabf7' : '#ff6b6b'; ctx.fill();
        // 繪製繩子
        ctx.beginPath(); ctx.moveTo(canvas.width/2, canvas.height/2 + radius);
        ctx.lineTo(canvas.width/2, canvas.height/2 + radius + 50);
        ctx.strokeStyle = '#666'; ctx.lineWidth = 3; ctx.stroke();
        requestAnimationFrame(draw);
    }
    draw();

    document.getElementById('startBtn').onclick = async function() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            analyser.fftSize = 256;
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            source.connect(analyser);

            this.style.display = 'none';
            mode = 'blowing';
            document.getElementById('status').innerText = "💨 請用力吹氣：「呼～～」";

            function process() {
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                // 我們只取中高頻部分，進一步過濾低頻雜音
                for(let i = 10; i < dataArray.length; i++) sum += dataArray[i];
                let avg = sum / (dataArray.length - 10);
                
                document.getElementById('debug').innerText = "環境音量: " + Math.floor(avg) + " (門檻: 100)";

                // 核心邏輯：音量必須超過 40 且處於吹氣模式
                if (mode === 'blowing') {
                    if (avg > 100) {
                        radius += 2.5; // 吹氣時長大
                    } else {
                        radius -= 0.5; // 沒吹時慢慢縮回
                    }
                    
                    if (radius >= targetRadius) {
                        mode = 'inhaling';
                        document.getElementById('status').innerText = "🌈 成功！現在請「慢慢吸氣」...";
                        document.getElementById('status').style.color = "#4dabf7";
                    }
                } else if (mode === 'inhaling') {
                    // 吸氣模式：固定節奏縮小，不受聲音影響
                    radius -= 0.4;
                    if (radius <= 50) {
                        mode = 'blowing';
                        document.getElementById('status').innerText = "💨 吐氣：再來一次！";
                        document.getElementById('status').style.color = "#ff4b4b";
                    }
                }

                radius = Math.max(50, Math.min(radius, 145));
                requestAnimationFrame(process);
            }
            process();
        } catch (err) {
            document.getElementById('status').innerText = "❌ 麥克風啟動失敗";
        }
    };
</script>
"""

# --- 2. 核心模型載入 ---
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

st.markdown("### 📝 第一步：請填寫基本資料")
c1, c2, c3 = st.columns(3)
with c1:
    grade = st.selectbox("年級", [f"{i}年級" for i in range(1, 7)])
with c2:
    classroom = st.selectbox("班級", [f"{i}班" for i in range(1, 7)])
with c3:
    student_name = st.text_input("名字", placeholder="小明")

st.divider()

st.markdown("### 🎤 第二步：對小精靈說說心情")
audio = mic_recorder(start_prompt="🎤 點我錄音", stop_prompt="🛑 錄音結束", key='recorder')

user_text = ""
if audio:
    with st.spinner('小精靈辨識中...'):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio['bytes'])
        result = stt_model.transcribe("temp_audio.wav", fp16=False)
        user_text = result['text']
        st.success(f"辨識結果：{user_text}")

manual_text = st.text_input("或是用打字的：", value=user_text)
final_text = manual_text if manual_text else user_text

st.markdown("### 🧐 第三步：你覺得現在心情如何?")
student_self_label = st.radio("心情選擇：", ["開心", "平淡", "難過", "生氣", "害怕", "驚奇"], horizontal=True)

if st.button("🚀 送出結果"):
    if final_text:
        prediction = emo_classifier(final_text)[0]
        ai_label = label_map.get(prediction['label'], "平淡")
        
        file_path = "student_logs.csv"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[now, grade, classroom, student_name, final_text, student_self_label, ai_label]], 
                                columns=["時間", "年級", "班級", "姓名", "內容", "自評情緒", "AI辨識情緒"])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        st.session_state['last_label'] = ai_label
        st.session_state['submitted'] = True
        st.rerun()

# --- 4. 回饋顯示 ---
if st.session_state.get('submitted'):
    label = st.session_state.get('last_label', '平淡')
    st.divider()
    st.subheader(f"小精靈覺得你現在：{label}")
    
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.info("🌟 氣球練習時間：對著麥克風吹氣讓氣球碰到虛線。")
        components.html(balloon_logic_html, height=600)
    else:
        st.balloons()
        st.success("紀錄成功!你今天很棒唷!")
    
    if st.button("關閉練習"):
        st.session_state['submitted'] = False
        st.rerun()

# --- 5. 管理側邊欄 ---
with st.sidebar:
    st.title("⚙️ 管理員選單")
    if st.checkbox("開啟數據查詢"):
        if os.path.exists("student_logs.csv"):
            try:
                df = pd.read_csv("student_logs.csv")
                st.dataframe(df)
                st.download_button("📥 下載數據", df.to_csv(index=False).encode('utf-8-sig'), "data.csv")
            except:
                st.error("資料格式有誤，請點下方重置。")
    
    if st.button("🚨 清空數據資料庫"):
        if os.path.exists("student_logs.csv"):
            os.remove("student_logs.csv")
            st.rerun()
