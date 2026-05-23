import streamlit as st
from src.predict import predict_sms


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Classifier")
st.markdown("Detect whether an SMS message is **Spam** or **Ham** in real time.")

st.divider()

message = st.text_area(
    "Enter your SMS message:",
    height=180,
    placeholder="Example: Congratulations! You have won ₹10,000..."
)

if st.button("Classify Message"):

    if not message.strip():
        st.warning("Please enter a message.")
    else:
        label, confidence = predict_sms(message)

        confidence = float(confidence) * 100

        if label == "SPAM":
            st.error(f"🚨 Prediction: {label}")
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            st.success(f"✅ Prediction: {label}")
            st.metric("Confidence", f"{confidence:.2f}%")