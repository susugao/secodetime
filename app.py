import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import whisper

# --- 1. 呼吸氣球 HTML (長吐短吸修正版) ---
balloon_logic_html = """
<div style="text-align: center; font-family: sans-serif; background: #ffffff; padding: 20px; border-radius: 20px;">
    <canvas id="balloonCanvas" width="300" height="400"></canvas>
    <div id="status" style="font-size: 22px; font-weight: bold; color: #ff4b4b; margin: 10px 0;">準備好練習呼吸了嗎?</div>
    <div id="debug" style="font-size: 12px; color: #ccc;">環境音量: 0</div>
    <button id="startBtn" style="padding: 15px 30px; font-size: 18px; background: #ff4b4b; color: white; border: none; border-radius: 50px; cursor: pointer;">🎤 開始深呼吸練習</button>
</div>

<script>
    var canvas = document.getElementById('balloonCanvas');
    var ctx = canvas.getContext('2d');
    var radius = 50;           
    var targetRadius = 50;     
    var mode = 'idle';         

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath(); 
        ctx.arc(canvas.width/2, canvas.height/2, targetRadius, 0, Math.PI * 2);
        ctx.setLineDash([5, 10]); 
        ctx.strokeStyle = '#ff4b4b'; 
        ctx.lineWidth = 3;
        ctx.stroke(); 
        ctx.setLineDash([]);
        ctx.beginPath(); 
        ctx.arc(canvas.width/2, canvas.height/2, radius, 0, Math.PI * 2);
        ctx.fillStyle = (mode === 'inhaling') ? '#4dabf7' : '#ff6b6b'; 
        ctx.fill();
        ctx.beginPath(); 
        ctx.moveTo(canvas.width/2, canvas.height/2 + radius);
        ctx.lineTo(canvas.width/2, canvas.height/2 + radius + 40);
        ctx.strokeStyle = '#666'; ctx.lineWidth = 2; ctx.stroke();
        requestAnimationFrame(draw);
    }
    draw();

    document.getElementById('startBtn').onclick = async function() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        analyser.fftSize = 256;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        source.connect(analyser);

        this.style.display = 'none';
        mode = 'blowing';

        function process() {
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for(let i = 10; i < dataArray.length; i++) sum += dataArray[i];
            let avg = sum / (dataArray.length - 10);
            document.getElementById('debug').innerText = "環境音量: " + Math.floor(avg);

            if (mode === 'blowing') {
                document.getElementById('status').innerText = "💨 慢慢吐氣：跟著虛線「呼～」";
                document.getElementById('status').style.color = "#ff4b4b";
                if (targetRadius < 130) targetRadius += 0.2; // 慢速吐氣
                if (avg > 30) { radius += 0.6; } else { radius -= 0.05; }
                if (radius >= targetRadius && targetRadius >= 125) { mode = 'inhaling'; }
            } else if (mode === 'inhaling') {
                document.getElementById('status').innerText = "🌈 輕鬆吸氣：準備下一次...";
                document.getElementById('status').style.color = "#4dabf7";
                if (targetRadius > 50) targetRadius -= 0.8; // 快速吸氣
                if (radius > 50) radius -= 0.8; // 快速吸氣
                if (radius <= 55 && targetRadius <= 55) { mode = 'blowing'; }
            }
            radius = Math.max(40, Math.min(radius, 150));
            requestAnimationFrame(process);
        }
        process();
    };
</script>
"""

# --- 2. 核心邏輯與模型 ---
st.set_page_config(page_title="情緒小精靈-呼吸練習版", page_icon="🌈")

@st.cache_resource
def load_models():
    emo_clf = pipeline("text-classification", model="Johnson8187/Chinese-Emotion-Small")
    stt_model = whisper.load_model("base")
    return emo_clf, stt_model

emo_classifier, stt_model = load_models()
label_map = {"LABEL_0": "平淡", "LABEL_1": "關切", "LABEL_2": "開心", "LABEL_3": "憤怒", 
             "LABEL_4": "悲傷", "LABEL_5": "疑問", "LABEL_6": "驚奇", "LABEL_7": "厭惡"}

# --- 3. 介面 ---
st.title("🌈 你的專屬情緒小精靈")

# 學生基本資料
st.markdown("### 📝 第一步：請確認個人資料")
c1, c2, c3 = st.columns(3)
with c1: grade = st.selectbox("年級", [f"{i}年級" for i in range(1, 7)])
with c2: classroom = st.selectbox("班級", [f"{i}班" for i in range(1, 7)])
with c3: student_name = st.text_input("名字", placeholder="小明")

st.divider()

# 錄音
st.markdown("### 🎤 第二步：對小精靈說說心情")
audio = mic_recorder(start_prompt="🎤 點我錄音", stop_prompt="🛑 錄完了", key='recorder')

user_text = ""
if audio:
    with st.spinner('小精靈正在聽...'):
        with open("temp_audio.wav", "wb") as f: f.write(audio['bytes'])
        result = stt_model.transcribe("temp_audio.wav", fp16=False)
        user_text = result['text']
        st.success(f"小精靈聽到了：{user_text}")

manual_text = st.text_input("或用打字的：", value=user_text)
final_text = manual_text if manual_text else user_text

st.markdown("### 🧐 第三步：你現在的心情是？")
student_self_label = st.radio("心情：", ["開心", "平淡", "難過", "生氣", "害怕", "驚奇"], horizontal=True)

if st.button("🚀 送出心情"):
    if final_text:
        prediction = emo_classifier(final_text)[0]
        ai_label = label_map.get(prediction['label'], "平淡")
        
        # 存檔
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([[now, grade, classroom, student_name, final_text, student_self_label, ai_label]], 
                                columns=["時間", "年級", "班級", "姓名", "內容", "自評情緒", "AI辨識情緒"])
        
        file_path = "student_logs.csv"
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_data.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        st.session_state['last_label'] = ai_label
        st.session_state['submitted'] = True
        st.rerun()

# --- 4. 練習區 ---
if st.session_state.get('submitted'):
    label = st.session_state.get('last_label', '平淡')
    st.divider()
    st.subheader(f"小精靈觀察到你現在的心情：{label}")
    
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.info(f"🌟 {student_name}，小精靈發現你有點不開心。讓我們跟著虛線，練習慢慢深呼吸，把不舒服吐出來。")
        components.html(balloon_logic_html, height=600)
        
        # 學生自覺練習完成按鈕
        if st.button("🧘 我覺得心情平靜了"):
            st.session_state['practice_done'] = True
            st.rerun()
    else:
        st.balloons()
        st.success(f"太棒了 {student_name}！希望你今天一直保持好心情！")
        if st.button("再說一次心情"):
            st.session_state['submitted'] = False
            st.rerun()

# --- 5. 鼓勵區 ---
if st.session_state.get('practice_done'):
    st.divider()
    st.balloons()
    st.markdown(f"""
    <div style="background-color: #e1f5fe; padding: 20px; border-radius: 15px; border-left: 5px solid #03a9f4;">
        <h2 style="color: #0288d1; margin-top: 0;">🌟 真的太棒了，{student_name}！</h2>
        <p style="font-size: 18px;">小精靈為你感到驕傲！你剛才很認真地練習呼吸，現在的你變得很冷靜，這就是自我照顧的力量喔！</p>
        <p style="font-size: 16px;">如果等一下心情又不好了，記得再來找小精靈練習深呼吸喔！🌈</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("重新開始"):
        st.session_state['submitted'] = False
        st.session_state['practice_done'] = False
        st.rerun()

# --- 6. 管理 ---
with st.sidebar:
    st.title("⚙️ 管理員選單")
    if st.checkbox("開啟數據模式"):
        if os.path.exists("student_logs.csv"):
            df = pd.read_csv("student_logs.csv")
            st.dataframe(df)
            st.download_button("📥 下載數據", df.to_csv(index=False).encode('utf-8-sig'), "data.csv")
    if st.button("🚨 清除所有紀錄"):
        if os.path.exists("student_logs.csv"): os.remove("student_logs.csv")
        st.rerun()
