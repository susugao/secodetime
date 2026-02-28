import streamlit as st
import streamlit.components.v1 as components

# --- 1. 定義 HTML/JS 氣球動畫組件 ---
# 這段代碼會建立一個畫布，並用 JavaScript 控制氣球放大縮小
balloon_interactive_html = """
<div style="text-align: center; font-family: sans-serif;">
    <canvas id="balloonCanvas" width="300" height="300" style="border: 1px solid #ddd; border-radius: 15px; background-color: #f9f9f9;"></canvas>
    <div style="margin-top: 15px;">
        <button id="breatheInBtn" style="padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">🌸 吸氣 (氣球變大)</button>
        <button id="breatheOutBtn" style="padding: 10px 20px; font-size: 16px; background-color: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px;">🍃 吐氣 (氣球變小)</button>
    </div>
    <p id="instructionText" style="font-size: 18px; color: #555; margin-top: 10px;">跟著小精靈一起做深呼吸吧！</p>
</div>

<script>
    const canvas = document.getElementById('balloonCanvas');
    const ctx = canvas.getContext('2d');
    const breatheInBtn = document.getElementById('breatheInBtn');
    const breatheOutBtn = document.getElementById('breatheOutBtn');
    const instructionText = document.getElementById('instructionText');

    // 氣球初始狀態
    let balloon = {
        x: canvas.width / 2,
        y: canvas.height / 2 + 30, // 稍微往下移，留出空間給結
        radius: 40,
        color: '#ff6b6b' // 溫暖的紅色
    };

    // 繪製氣球的函數
    function drawBalloon(r) {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // 清空畫布

        // 繪製氣球主體
        ctx.beginPath();
        ctx.arc(balloon.x, balloon.y, r, 0, Math.PI * 2);
        ctx.fillStyle = balloon.color;
        ctx.fill();
        ctx.closePath();

        // 繪製氣球的結
        ctx.beginPath();
        ctx.moveTo(balloon.x, balloon.y + r);
        ctx.lineTo(balloon.x - 10, balloon.y + r + 15);
        ctx.lineTo(balloon.x + 10, balloon.y + r + 15);
        ctx.fillStyle = balloon.color;
        ctx.fill();
        ctx.closePath();
    }

    // 初始繪製
    drawBalloon(balloon.radius);

    // --- 互動邏輯 ---
    let targetRadius = 40;
    const minRadius = 40;
    const maxRadius = 100;
    const animationSpeed = 2; // 調整動畫流暢度

    function animate() {
        if (balloon.radius < targetRadius) {
            balloon.radius += animationSpeed;
            if (balloon.radius > targetRadius) balloon.radius = targetRadius;
        } else if (balloon.radius > targetRadius) {
            balloon.radius -= animationSpeed;
            if (balloon.radius < targetRadius) balloon.radius = targetRadius;
        }

        drawBalloon(balloon.radius);

        if (balloon.radius !== targetRadius) {
            requestAnimationFrame(animate);
        }
    }

    // 點擊吸氣按鈕
    breatheInBtn.addEventListener('click', () => {
        targetRadius = maxRadius;
        instructionText.innerText = "🌸 慢慢吸氣... 感覺小腹變大... (氣球也變大囉)";
        instructionText.style.color = "#4CAF50";
        animate();
    });

    // 點擊吐氣按鈕
    breatheOutBtn.addEventListener('click', () => {
        targetRadius = minRadius;
        instructionText.innerText = "🍃 慢慢吐氣... 把煩惱都吐出來... (氣球縮小囉)";
        instructionText.style.color = "#2196F3";
        animate();
    });
</script>
"""
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
    
    # 3. 互動任務 (當情緒不穩定時)
    if label in ["憤怒", "悲傷", "厭惡"]:
        with st.expander("🌟 點開這裡，小精靈陪你做個練習"):
            st.write("1. 請先閉上眼睛。")
            st.write("2. 慢慢吸氣... 1, 2, 3...")
            st.write("3. 慢慢吐氣... 1, 2, 3...")
            if st.button("我做完了，感覺好一點了"):
                st.success("你真棒！自我覺察第一步完成！")
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
