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
