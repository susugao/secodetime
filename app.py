import streamlit as st
import streamlit.components.v1 as components

# --- 1. 定義 HTML/JS 氣球動畫組件 ---
# 這段代碼會建立一個畫布，並用 JavaScript 控制氣球放大縮小
# --- 替換為：麥克風感測氣球 HTML/JS 定義 ---
balloon_interactive_html = """
<div style="text-align: center; font-family: sans-serif;">
    <canvas id="balloonCanvas" width="300" height="250" style="border-radius: 15px; background-color: #f0f2f6;"></canvas>
    <div id="status" style="margin-top: 10px; font-weight: bold; color: #ff4b4b;">點擊下方按鈕開啟感應</div>
    <button id="startMic" style="padding: 10px 20px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer;">🎤 開始練習 (請對著麥克風吹氣)</button>
    <p id="instruction" style="font-size: 16px; color: #555; margin-top: 10px;">吸氣時暫停，吐氣時用力對著麥克風「呼～」</p>
</div>

<script>
    const canvas = document.getElementById('balloonCanvas');
    const ctx = canvas.getContext('2d');
    const statusText = document.getElementById('status');
    const startBtn = document.getElementById('startMic');

    let balloonRadius = 50;
    let audioContext, analyser, microphone, javascriptNode;

    function draw(r) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, r, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b6b';
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(canvas.width/2, canvas.height/2 + r);
        ctx.lineTo(canvas.width/2, canvas.height/2 + r + 30);
        ctx.strokeStyle = '#555';
        ctx.stroke();
    }

    draw(balloonRadius);

    startBtn.onclick = function() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
            statusText.innerText = "🌟 感應中！請開始深呼吸...";
            statusText.style.color = "#28a745";
            startBtn.style.display = "none";

            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            microphone = audioContext.createMediaStreamSource(stream);
            javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);

            analyser.smoothingTimeConstant = 0.8;
            analyser.fftSize = 1024;

            microphone.connect(analyser);
            analyser.connect(javascriptNode);
            javascriptNode.connect(audioContext.destination);

            javascriptNode.onaudioprocess = function() {
                let array = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(array);
                let values = 0;
                for (let i = 0; i < array.length; i++) { values += array[i]; }
                let average = values / array.length; 

                if (average > 25) {  // 靈敏度門檻
                    if (balloonRadius < 110) balloonRadius += 2; 
                } else {
                    if (balloonRadius > 50) balloonRadius -= 0.8;
                }
                draw(balloonRadius);
            };
        }).catch(function(err) {
            statusText.innerText = "無法開啟麥克風，請檢查權限設定";
        });
    };
</script>
"""
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
# --- 增加這段回饋字典 ---
feedback_dict = {
    "開心": {
        "msg": ["哇！你的好心情像陽光一樣閃亮！", "聽起來今天有很棒的事情發生呢！", "太好了，小精靈也想跟你一起跳舞！"],
        "action": "balloons" # 噴氣球
    },
    "憤怒": {
        "msg": ["小精靈感覺到你心裡火辣辣的。", "沒關係，生氣是正常的。我們先停下來，喝口水好嗎？", "呼～我們要不要一起把氣球吹大，然後放掉？"],
        "action": "warning" # 顯示警告顏色
    },
    "悲傷": {
        "msg": ["沒關係，想哭的時候小精靈會陪著你。", "抱一個！你今天辛苦了。", "如果你覺得累，休息一下也很好喔。"],
        "action": "info"
    },
    "平淡": {
        "msg": ["平穩的一天也是很珍貴的喔。", "嗯嗯，這是一個適合觀察世界的好時機。", "心情安安靜靜的，感覺很舒服呢！"],
        "action": "snow" # 飄雪（代表平靜）
    }
}

# --- 在情緒辨識後的顯示部分 ---
if final_text:
    prediction = emo_classifier(final_text)[0]
    label = label_map.get(prediction['label'], "平淡")
    
    # 隨機挑選一句回饋
    import random
    fb = feedback_dict.get(label, {"msg": ["謝謝你告訴我你的感受。"], "action": None})
    message = random.choice(fb["msg"])
    
    st.divider()
    
    # 1. 視覺回饋
    if fb["action"] == "balloons":
        st.balloons()
    elif fb["action"] == "snow":
        st.snow()
        
    # 2. 顯示對話框
    st.chat_message("assistant", avatar="🌈").write(f"**小精靈說：** {message}")
    
# --- 在情緒分析與視覺回饋之後 ---
# 假設你之前的 label 翻譯字典與回饋字典都還在

if final_text:
    prediction = emo_classifier(final_text)[0]
    label = label_map.get(prediction['label'], "平淡")
    
    # ... (之前的 Emoji 顯示與 random message 顯示) ...

    # 4. 關鍵介入：當偵測到負面情緒時，彈出視覺化呼吸練習
    if label in ["憤怒", "悲傷", "厭惡"]:
        st.divider()
        st.subheader("🌈 小精靈陪你做個『冷靜氣球』練習")
        st.write("請跟著下方的氣球，一起慢慢呼吸三次。")
        
        # 嵌入剛才定義的 HTML/JS 組件
        components.html(balloon_interactive_html, height=450)
        
        # 增加一個完成按鈕，強化正向回饋
        if st.button("我做完了三次呼吸，感覺好一點了！"):
            st.balloons()
            st.success("你真棒！成功找回平靜的能量囉！")
