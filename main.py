import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Load NLTK data
nltk.download('stopwords')

ps = PorterStemmer()

def text_process(mess):
    """
    Takes in a string of text, performs the following:
    1. Make text lowercase
    2. Remove text in square brackets
    3. Remove links
    4. Remove punctuation
    5. Remove words containing numbers
    6. Remove stopwords
    Returns a cleaned string of text
    """
    text = str(mess).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets

    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub('<.*?>+', '', text)
    
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    
    # Remove punctuation
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', nopunc)
    
    # Remove stopwords
    return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])

stemmer = nltk.SnowballStemmer("english")

def preprocess_data(text):
    # Clean punctuation, URLs, and so on
    text = text_process(text)
    # Stem all words in the sentence
    return ' '.join(stemmer.stem(word) for word in text.split(' '))

# Load pre-trained model
model = pickle.load(open('MultinomialNB_model.pkl', 'rb'))

# Add background CSS for customization
st.markdown(
    """
    <style>
    .main {
        background-color: #B0B0B0;  /* Change the color here */
    }
    .stTextInput, .stTextArea {
        background-color: #003366; /* Light blue background for input */
        color: #4CAF50;  /* Dark blue text color */
        border: 2px solid #4CAF50; /* Green border */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green background for button */
        color: Dark blue; /* White text color */
        border: none; /* No border */
        padding: 10px 20px; /* Padding for button */
        border-radius: 5px; /* Rounded corners */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title with a larger font
st.markdown("<h1 style='text-align: center; color:#003366;'>Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Add description
st.markdown("""
<div style='color: #003366; text-align: center;'>  <!-- Dark blue color -->
### 🌟 <strong>Instructions:</strong>
- Enter a text message or email content.<br>
- Click on <strong>'Predict'</strong> to classify whether it is Spam or Not Spam.
</div>
""", unsafe_allow_html=True)

# Input text box with a placeholder
input_sms = st.text_area("Enter the message", placeholder="Type your email or SMS here...", height=150)

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Predict button
if st.button('Predict'):
    
    if input_sms.strip() != "":  # Check for empty input

        # 1. Preprocess the input
        transformed_sms = preprocess_data(input_sms)

        # 2. Make prediction
        result = model.predict([transformed_sms])[0]

        # 3. Display the result with color and icons
        if result == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>🚨 This message is Spam 🚨</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>✅ This message is Not Spam ✅</h2>", unsafe_allow_html=True)

    else:
        st.markdown("<p style='color: red;'>❌ Please enter a valid message to classify.</p>", unsafe_allow_html=True)
