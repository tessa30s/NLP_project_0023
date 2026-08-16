import streamlit as st
import pickle
import re
import os
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# NLTK SETUP
# ============================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "models", "gru_model.keras")
)

TOKENIZER_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ GRU model not found:\n{MODEL_PATH}")
    st.stop()

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"❌ Tokenizer not found:\n{TOKENIZER_PATH}")
    st.stop()


# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

@st.cache_resource
def load_resources():

    loaded_model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        loaded_tokenizer = pickle.load(file)

    return loaded_model, loaded_tokenizer


model, tokenizer = load_resources()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split into words
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>

/* ============================================================
   GLOBAL PAGE
   ============================================================ */

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(101, 66, 190, 0.18),
            transparent 28%
        ),

        radial-gradient(
            circle at 90% 10%,
            rgba(65, 92, 190, 0.15),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #071126 0%,
            #0b1228 50%,
            #080d1d 100%
        );

    color: #ffffff;
}


/* ============================================================
   REMOVE STREAMLIT DEFAULT ELEMENTS
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ============================================================
   MAIN CONTAINER
   ============================================================ */

.block-container {

    max-width: 1180px;

    padding-top: 1rem;
    padding-bottom: 2rem;
}


/* ============================================================
   TOP NAVIGATION
   ============================================================ */

.top-nav {

    height: 65px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0 15px;

    border-bottom:
        1px solid rgba(145, 125, 255, 0.16);

    margin-bottom: 30px;
}


.brand {

    display: flex;

    align-items: center;

    gap: 12px;

    font-size: 21px;

    font-weight: 700;

    color: #f5f3ff;
}


.brand-icon {

    width: 43px;

    height: 43px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #693be8,
            #a54de5
        );

    box-shadow:
        0 0 25px
        rgba(130, 70, 240, 0.35);

    font-size: 23px;
}


.deploy-button {

    padding: 10px 20px;

    border-radius: 11px;

    background:
        linear-gradient(
            135deg,
            #693ce8,
            #a343d7
        );

    color: white;

    font-weight: 700;

    box-shadow:
        0 8px 25px
        rgba(115, 61, 231, 0.25);
}


/* ============================================================
   HERO SECTION
   ============================================================ */

.hero {

    text-align: center;

    margin-top: 10px;

    margin-bottom: 28px;
}


.hero-icon {

    width: 64px;

    height: 64px;

    margin: 0 auto;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 18px;

    border:
        1px solid
        rgba(184, 92, 255, 0.65);

    background:
        linear-gradient(
            145deg,
            rgba(86, 67, 160, 0.35),
            rgba(33, 48, 102, 0.5)
        );

    box-shadow:
        0 0 30px
        rgba(136, 73, 255, 0.18);

    font-size: 31px;
}


.hero-title {

    font-size: 47px;

    font-weight: 800;

    margin-top: 12px;

    margin-bottom: 5px;

    background:
        linear-gradient(
            90deg,
            #d866ff,
            #8c9cff,
            #67a9ff
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.hero-subtitle {

    font-size: 17px;

    color: #c8cde0;

    margin-top: 5px;
}


.spam-word {

    color: #ff466b;

    font-weight: 700;
}


.ham-word {

    color: #4ee69b;

    font-weight: 700;
}


/* ============================================================
   GLASS CARD
   ============================================================ */

.glass-card {

    background:
        linear-gradient(
            145deg,
            rgba(29, 35, 67, 0.82),
            rgba(15, 22, 46, 0.88)
        );

    border:
        1px solid
        rgba(148, 87, 255, 0.48);

    border-radius: 17px;

    padding: 25px;

    box-shadow:
        0 20px 50px
        rgba(0, 0, 0, 0.25);

    margin-bottom: 18px;
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 18px;

    font-weight: 700;

    color: #f3f3ff;

    margin-bottom: 14px;
}


/* ============================================================
   TEXT AREA
   ============================================================ */

textarea {

    background:
        rgba(5, 12, 30, 0.82) !important;

    border:
        1px solid
        rgba(110, 117, 185, 0.45) !important;

    border-radius: 12px !important;

    color: white !important;

    font-size: 16px !important;
}


textarea::placeholder {

    color: #7f88a5 !important;
}


textarea:focus {

    border:
        1px solid
        #9b57ff !important;

    box-shadow:
        0 0 0 2px
        rgba(155, 87, 255, 0.15) !important;
}


/* ============================================================
   HIDE TEXTAREA LABEL
   ============================================================ */

div[data-testid="stTextArea"] label {

    display: none;
}


/* ============================================================
   CHARACTER COUNT
   ============================================================ */

.char-count {

    color: #858eac;

    font-size: 13px;

    margin-top: 7px;

    margin-bottom: 3px;
}


/* ============================================================
   PREDICT BUTTON
   ============================================================ */

div.stButton {

    display: flex;

    justify-content: center;

    margin-top: 13px;
}


div.stButton > button {

    border: none;

    border-radius: 10px;

    padding: 11px 32px;

    font-size: 16px;

    font-weight: 700;

    color: white;

    background:
        linear-gradient(
            135deg,
            #7139ed,
            #b044d8
        );

    box-shadow:
        0 8px 25px
        rgba(118, 57, 237, 0.30);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


div.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 30px
        rgba(144, 65, 255, 0.45);

    color: white;
}


/* ============================================================
   PREDICTION LABEL
   ============================================================ */

.probability-label {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-top: 13px;

    margin-bottom: 6px;

    font-size: 15px;

    color: #d9dced;
}


.spam-percent {

    color: #ff466b;

    font-weight: 700;
}


.ham-percent {

    color: #36dc8d;

    font-weight: 700;
}


/* ============================================================
   PROGRESS BACKGROUND
   ============================================================ */

.progress-background {

    width: 100%;

    height: 9px;

    background: #101a32;

    border-radius: 20px;

    overflow: hidden;

    margin-bottom: 17px;
}


/* ============================================================
   SPAM PROGRESS
   ============================================================ */

.spam-progress {

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #ff385d,
            #ff5575
        );

    border-radius: 20px;

    transition: width 0.5s ease;
}


/* ============================================================
   HAM PROGRESS
   ============================================================ */

.ham-progress {

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #35d98b,
            #55eca4
        );

    border-radius: 20px;

    transition: width 0.5s ease;
}


/* ============================================================
   RESULT BOX
   ============================================================ */

.result-box {

    min-height: 145px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    border-radius: 13px;

    padding: 20px;
}


.result-spam {

    background:
        linear-gradient(
            145deg,
            rgba(113, 35, 61, 0.35),
            rgba(70, 28, 58, 0.45)
        );

    border:
        1px solid
        rgba(255, 65, 105, 0.55);
}


.result-ham {

    background:
        linear-gradient(
            145deg,
            rgba(24, 91, 70, 0.35),
            rgba(21, 71, 61, 0.45)
        );

    border:
        1px solid
        rgba(58, 224, 145, 0.50);
}


.result-icon {

    font-size: 38px;

    margin-bottom: 5px;
}


.result-title {

    font-size: 23px;

    font-weight: 800;

    margin-bottom: 5px;
}


.spam-title {

    color: #ff4770;
}


.ham-title {

    color: #41e29a;
}


.result-description {

    color: #c9cce0;

    font-size: 14px;
}


/* ============================================================
   CONFIDENCE
   ============================================================ */

.confidence-box {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-top: 16px;

    padding: 14px 18px;

    border-radius: 11px;

    background:
        rgba(19, 28, 55, 0.85);

    border:
        1px solid
        rgba(118, 130, 180, 0.18);
}


.confidence-label {

    color: #e4e6f3;

    font-weight: 600;
}


.confidence-value {

    color: #b58cff;

    font-weight: 800;
}


/* ============================================================
   FOOTER
   ============================================================ */

.info-footer {

    text-align: center;

    padding: 14px;

    border-radius: 11px;

    background:
        rgba(18, 27, 55, 0.75);

    border:
        1px solid
        rgba(120, 100, 220, 0.28);

    color: #aeb5d1;

    font-size: 14px;

    margin-top: 15px;
}


/* ============================================================
   WARNING / ERROR / SUCCESS
   ============================================================ */

div[data-testid="stAlert"] {

    border-radius: 11px;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .hero-title {

        font-size: 35px;
    }

    .hero-subtitle {

        font-size: 15px;
    }

    .glass-card {

        padding: 18px;
    }

    .top-nav {

        padding: 0 5px;
    }

}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown(
"""
<div class="top-nav">

<div class="brand">

<div class="brand-icon">
💬
</div>

<span>SMS Spam Detection</span>

</div>

<div class="deploy-button">
🚀 Deploy
</div>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero">

<div class="hero-icon">
✉️
</div>

<div class="hero-title">
SMS Spam Detection
</div>

<div class="hero-subtitle">
Enter an SMS message below to check whether it is
<span class="spam-word">Spam</span>
or
<span class="ham-word">Ham</span>.
</div>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# INPUT CARD - OPEN
# ============================================================

st.markdown(
"""
<div class="glass-card">

<div class="section-title">
🔍
<span>Enter SMS Message</span>
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Message",
    placeholder="Type or paste your SMS message here...",
    height=125,
    max_chars=1000,
    label_visibility="collapsed"
)


# ============================================================
# CHARACTER COUNT
# ============================================================

character_count = len(message)

st.markdown(
f"""
<div class="char-count">
{character_count} / 1000 characters
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "✨  Predict"
)


# ============================================================
# INPUT CARD - CLOSE
# ============================================================

st.markdown(
"""
</div>
""",
unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if message.strip() == "":

        st.warning(
            "⚠️ Please enter an SMS message first."
        )

    else:

        # ====================================================
        # PREPROCESS MESSAGE
        # ====================================================

        cleaned_message = preprocess_text(message)


        # ====================================================
        # TOKENIZE
        # ====================================================

        sequence = tokenizer.texts_to_sequences(
            [cleaned_message]
        )


        # ====================================================
        # PAD SEQUENCE
        # ====================================================

        padded_sequence = pad_sequences(
            sequence,
            maxlen=100,
            padding="post",
            truncating="post"
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            padded_sequence,
            verbose=0
        )


        spam_probability = float(
            prediction[0][0]
        )


        # Make sure probability stays between 0 and 1
        spam_probability = max(
            0.0,
            min(1.0, spam_probability)
        )


        ham_probability = (
            1.0 - spam_probability
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if spam_probability >= 0.5:

            result = "Spam"

            confidence = spam_probability

        else:

            result = "Ham"

            confidence = ham_probability


        # ====================================================
        # RESULT CARD - OPEN
        # ====================================================

        st.markdown(
        """
        <div class="glass-card">

        <div class="section-title">
        📊
        <span>Prediction Result</span>
        </div>
        """,
        unsafe_allow_html=True
        )


        # ====================================================
        # TWO COLUMNS
        # ====================================================

        left_column, right_column = st.columns(
            [1.05, 0.95],
            gap="large"
        )


        # ====================================================
        # PROBABILITY SECTION
        # ====================================================

        with left_column:

            # ------------------------------------------------
            # SPAM
            # ------------------------------------------------

            st.markdown(
            f"""
            <div class="probability-label">

            <span>
            📨 Spam Probability
            </span>

            <span class="spam-percent">
            {spam_probability:.2%}
            </span>

            </div>

            <div class="progress-background">

            <div
            class="spam-progress"
            style="width:{spam_probability * 100:.2f}%">
            </div>

            </div>
            """,
            unsafe_allow_html=True
            )


            # ------------------------------------------------
            # HAM
            # ------------------------------------------------

            st.markdown(
            f"""
            <div class="probability-label">

            <span>
            📩 Ham Probability
            </span>

            <span class="ham-percent">
            {ham_probability:.2%}
            </span>

            </div>

            <div class="progress-background">

            <div
            class="ham-progress"
            style="width:{ham_probability * 100:.2f}%">
            </div>

            </div>
            """,
            unsafe_allow_html=True
            )


        # ====================================================
        # RESULT SECTION
        # ====================================================

        with right_column:

            if result == "Spam":

                st.markdown(
                """
                <div class="result-box result-spam">

                <div class="result-icon">
                🚨
                </div>

                <div class="result-title spam-title">
                Spam Message
                </div>

                <div class="result-description">
                This message is classified as SPAM.
                </div>

                </div>
                """,
                unsafe_allow_html=True
                )

            else:

                st.markdown(
                """
                <div class="result-box result-ham">

                <div class="result-icon">
                ✅
                </div>

                <div class="result-title ham-title">
                Ham Message
                </div>

                <div class="result-description">
                This message is classified as HAM.
                </div>

                </div>
                """,
                unsafe_allow_html=True
                )


        # ====================================================
        # CONFIDENCE SCORE
        # ====================================================

        st.markdown(
        f"""
        <div class="confidence-box">

        <span class="confidence-label">
        🛡️ Confidence Score
        </span>

        <span class="confidence-value">
        {confidence:.2%}
        </span>

        </div>
        """,
        unsafe_allow_html=True
        )


        # ====================================================
        # RESULT CARD - CLOSE
        # ====================================================

        st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
        )


# ============================================================
# INFORMATION FOOTER
# ============================================================

st.markdown(
"""
<div class="info-footer">
ⓘ This model uses Machine Learning to analyze and classify SMS messages.
</div>
""",
unsafe_allow_html=True
)