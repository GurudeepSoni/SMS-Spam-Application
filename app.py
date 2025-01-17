import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure necessary NLTK resources are downloaded (including punkt tokenizer and stopwords)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Load pre-trained vectorizer and model
try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure they are in the correct directory.")
    st.stop()

# Streamlit UI Components
st.title("SMS Spam Detection Model")
st.write("*This is a Machine Learning project that uses Natural Language Processing to classify SMS messages as spam or not.*")

# Create a text input for SMS
input_sms = st.text_input("Enter the SMS", help="Type your message here...", max_chars=300)

# Add a button to predict
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter an SMS to classify.")
    else:
        with st.spinner('Processing...'):
            # Preprocess the SMS input
            transformed_sms = transform_text(input_sms)

            # Vectorize the input text
            vector_input = tk.transform([transformed_sms])

            # Predict the result (Spam or Not Spam)
            result = model.predict(vector_input)[0]

            # Display the result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")

# Custom CSS (optional, adjust as needed)
st.markdown("""
    <link rel="stylesheet" href="styles.css">
    <div class="container">
        <h1 class="title">SMS Spam Detection Model 🌟</h1>
        <p class="description">This is a Machine Learning project that uses Natural Language Processing to classify SMS messages as spam or not. 🚀📱</p>
    </div>
""", unsafe_allow_html=True)

# Example SMS for user guidance
st.text_area('Example SMS', "Free money! Win a lottery today. Text WIN to 12345.")

# Display watermark
st.markdown('<div class="watermark" style="color: #bbb; font-style: italic; opacity: 0.5; font-size: 14px;">Created by Gurudeep Soni 💻</div>', unsafe_allow_html=True)
