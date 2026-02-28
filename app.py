import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper
import random

# --- 1. 呼吸氣球 HTML/JS (含精準頻率過濾與吹吸循環) ---
balloon_logic_html = """
<div style="text-align: center; font-family: sans-serif; background: #ffffff; padding: 10px; border-radius: 20px;">
    <canvas id="balloonCanvas" width="300" height="400"></canvas>
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
    let mode = 'idle'; 
    let smoothedVolume = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 1. 目標虛線
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, targetRadius, 0, Math.PI * 2);
        ctx.setLineDash([5, 8]);
        ctx.strokeStyle = '#dddddd';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. 氣球
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, radius, 0, Math.PI * 2);
        ctx.fillStyle = (mode === 'inhaling') ? '#4dabf7' : '#ff6b6b';
        ctx.fill();
        
        // 3. 繩子
        ctx.beginPath();
        ctx.moveTo(canvas.width/2, canvas.height/2 + radius);
        ctx.lineTo(canvas.width/2, canvas.height/2 + radius + 50);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 3;
        ctx.stroke();

        requestAnimationFrame(draw);
    }

    startBtn.onclick = async function() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            analyser.fftSize = 512;
            source.connect(analyser);

            startBtn.style.display = 'none';
            mode = 'blowing';
            statusText.innerText = "💨 吐氣：對著麥克風「呼～」";

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            function process() {
                analyser.getByteFrequencyData(dataArray);
                let blowingEnergy = 0;
                let speechEnergy = 0;
                
                for(let i = 0; i < dataArray.length; i++) {
                    if (i > 25 && i < 120) blowingEnergy += dataArray[i]; // 偵測中高頻吹氣
                    if (i <= 25) speechEnergy += dataArray[i];            // 背景低頻音
                }
                
                let avgBlowing = blowingEnergy / 95;
                let avgSpeech = speechEnergy / 25;

                // 排除說話與背景音：吹氣能量需顯著高於低頻音
                if (avgBlowing > 15 && avgBlowing > avgSpeech * 1.1){
                    smoothedVolume = smoothedVolume * 0.7 + avgBlowing * 0.3;
                } else {
                    smoothedVolume = smoothedVolume * 0.8;
                }

                if (mode === 'blowing') {
                    if (smoothedVolume > 25) {
                        radius += 1.2;
                    } else {
                        radius -= 0.2;
                    }
                    if (radius >= targetRadius) {
                        mode = 'inhaling';
                        statusText.innerText = "🌈 成功！現在請「慢慢吸氣」...";
                        statusText.style.color = "#4dabf7";
                    }
                } else if (mode === 'inhaling') {
                    radius -= 0.35; 
                    if (radius <= 55) {
                        mode = 'blowing';
                        statusText.innerText = "💨 吐氣：再來一次「呼～」";
                        statusText.style.color = "#ff4b4b";
                    }
                }
                radius = Math.max(50, Math.min(radius, 145));
                requestAnimationFrame(process);
            }
            process();
        } catch (err) {
            statusText.innerText = "❌ 麥克風權限未開啟";
        }
    };
    draw();
</script>
"""

# --- 2. 核心模型與存檔設定 ---
st.set_page_config(page_title="情緒小精靈-研究版", page_icon="🌈")

@st.cache_resource
def load_models():
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

# --- 3. 學生操作介面 ---
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

st.markdown("### 🧐 第三步：你覺得你現在的心情是？")
student_self_label = st.radio("選擇心情：", ["開心", "平淡", "難過", "生氣", "害怕", "驚奇"], horizontal=True)

if st.button("🚀 點我送出給小精靈"):
    if final_text:
        prediction = emo_classifier(final_text)[0]
        ai_label = label_map.get(prediction['label'], "平淡")
        
        # 存檔邏輯 (7個欄位齊全)
        file_path = "student_logs.csv"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[now, grade, classroom, student_name, final_text, student_self_label, ai_label]], 
                                columns=["時間", "年級", "班級", "姓名", "內容", "自評情緒", "AI辨識情緒"])
        
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        # 設定回饋狀態
        st.session_state['last_label'] = ai_label
        st.session_state['submitted'] = True
        st.rerun() # 送出後重置介面，避免重複偵測

# --- 4. 互動回饋區 (提交後顯示) ---
if st.session_state.get('submitted'):
    label = st.session_state.get('last_label', '平淡')
    st.divider()
    st.subheader(f"分析結果：{label}")
    
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.info("🌟 小精靈陪你練習深呼吸：吹氣碰到虛線，然後慢慢吸氣。")
        components.html(balloon_logic_html, height=600)
    else:
        st.balloons()
        st.success("紀錄成功！你今天做得很棒！")
    
    if st.button("完成練習"):
        st.session_state['submitted'] = False
        st.rerun()

# --- 5. 管理後台 ---
with st.sidebar:
    st.title("⚙️ 研究管理")
    if st.checkbox("開啟數據模式"):
        file_path = "student_logs.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                st.dataframe(df)
                st.download_button("📥 下載研究數據", df.to_csv(index=False).encode('utf-8-sig'), "data.csv")
            except:
                st.error("偵測到舊格式衝突，請點擊下方重置按鈕。")
        else:
            st.info("目前尚無數據。")
    
    st.divider()
    if st.button("🚨 強制重置資料庫 (清除紅字)"):
        if os.path.exists("student_logs.csv"):
            os.remove("student_logs.csv")
            st.rerun()
