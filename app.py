import streamlit as st
import pickle

# Load the trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure they are in the correct directory.")
    st.stop()

# Streamlit title and description
st.title("📱 SMS Spam Classification Application 🚀")
st.write("✨ This is a Machine Learning application to classify SMS messages as **spam** or **not spam**. 🌟")

# Input text area for user to enter SMS content
user_input = st.text_area("📝 Enter an SMS to classify:", height=150)

# Button to classify the SMS
if st.button("🎯 Classify"):
    if user_input.strip():  # Check if the input is not empty
        # Preprocess and classify the SMS
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)

        # Display the result
        if result[0] == 0:
            st.success("✅ The SMS is **not spam**. 📩")
        else:
            st.error("❌ The SMS is **spam**. 🛑")
    else:
        st.warning("⚠️ Please type an SMS to classify.")

# Example SMS (Read-only Text Area for reference)
st.subheader("💡 Example SMS:")
st.text("Congratulations! You've won a $1000 gift card. Click here to claim your prize now.")

# Add a watermark using HTML and CSS
st.markdown("""
    <style>
    .watermark {
        color: #bbb;
        font-style: italic;
        opacity: 0.7;
        font-size: 14px;
        text-align: center;
        margin-top: 50px;
    }
    .title {
        color: #4CAF50;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    <div class="watermark">
        Created by Gurudeep Soni 💻
    </div>
""", unsafe_allow_html=True)
