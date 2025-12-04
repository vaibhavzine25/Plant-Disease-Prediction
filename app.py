import os
from io import BytesIO

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Guruprasad Krushiseva Kendra",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for languages & page
if "language" not in st.session_state:
    st.session_state.language = "english"
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #f7f9fa; }
    h1 { 
        text-align: center; 
        color: #1b4332; 
        margin-bottom: 0; 
        font-family: 'Arial', sans-serif;
        line-height: 1.2;
        word-wrap: break-word;
        padding: 0 10px;
    }
    .subtext { 
        text-align: center; 
        color: #4b5563; 
        margin-top: 10px;
        line-height: 1.4;
        padding: 0 20px;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Sidebar Styling */
    .css-1d391kg { padding-top: 1rem; }
    .sidebar .sidebar-content { background-color: #f0f8f0; }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
        .stSidebar > div:first-child { width: 300px; }
        .css-1d391kg { padding: 1rem 0.5rem; }
        .block-container { padding: 1rem; }
        h1 { 
            font-size: 1.8rem;
            line-height: 1.3;
            padding: 0 5px;
        }
        .subtext { 
            font-size: 0.9rem;
            padding: 0 10px;
        }
    }
    
    @media (max-width: 480px) {
        .stSidebar > div:first-child { width: 280px; }
        h1 { 
            font-size: 1.5rem;
            line-height: 1.4;
            padding: 0 5px;
        }
        .css-1d391kg { padding: 0.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "EfficientNetB0_plant_disease.keras")
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ============================================================
# LANGUAGE OPTIONS & TRANSLATIONS
# ============================================================

ui_languages = {
    "English": "en",
    "मराठी": "mr",
}

detection_languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
}

translations = {
    "english": {
        "title": "Guruprasad Krushiseva Kendra",
        "subtitle": "Upload or capture a plant image to detect disease and receive treatment suggestions.",
        "ui_language": "UI Language / यूआय भाषा:",
        "detection_language": "Detection & Audio Language:",
        "navigation": "Navigation",
        "select_page": "Select Page:",
        "home": "Home",
        "about": "About Us",
        "contact": "Contact",
        "interface_language": "Interface Language",
        "contact_info": "Contact Info",
        "phone": "Phone:",
        "address": "Address:",
        "email": "Email:",
        "quick_stats": "Quick Stats",
        "crops_supported": "10 Crops Supported",
        "languages_available": "10 Languages Available",
        "ai_powered": "AI-Powered Detection",
        "upload_gallery": "Upload from Gallery",
        "capture_camera": "Capture from Camera",
        "choose_image": "Choose a leaf image",
        "take_photo": "Take a photo",
        "selected_image": "Selected Image",
        "diagnose": "Diagnose",
        "analyzing": "Analyzing image...",
        "disease_detected_en": "Disease Detected (English):",
        "disease_detected_lang": "Disease Detected",
        "suggestion_en": "Suggestion (English)",
        "suggestion_lang": "Suggestion",
        "audio_success": "Audio generated successfully!",
        "audio_failed": "Audio generation failed, but text results are available above.",
        "translation_error": "Translation unavailable. Please read English suggestion below.",
        "non_plant_error": "This image does not look like a plant leaf. Please upload a clear leaf image.",
    },
    "marathi": {
        "title": "गुरुप्रसाद कृषि सेवा केंद्र",
        "subtitle": "रोग शोधण्यासाठी आणि उपचार सूचना मिळविण्यासाठी वनस्पतीची प्रतिमा अपलोड करा किंवा कॅप्चर करा.",
        "ui_language": "यूआई भाषा / UI Language:",
        "detection_language": "निदान आणि ऑडिओ भाषा:",
        "navigation": "नेव्हिगेशन",
        "select_page": "पृष्ठ निवडा:",
        "home": "मुख्यपृष्ठ",
        "about": "आमच्याबद्दल",
        "contact": "संपर्क",
        "interface_language": "इंटरफेस भाषा",
        "contact_info": "संपर्क माहिती",
        "phone": "फोन:",
        "address": "पत्ता:",
        "email": "ईमेल:",
        "quick_stats": "द्रुत आकडेवारी",
        "crops_supported": "10 पिके समर्थित",
        "languages_available": "10 भाषा उपलब्ध",
        "ai_powered": "AI-संचालित शोध",
        "upload_gallery": "गॅलरीमधून अपलोड करा",
        "capture_camera": "कॅमेऱ्यातून कॅप्चर करा",
        "choose_image": "पानाची प्रतिमा निवडा",
        "take_photo": "फोटो घ्या",
        "selected_image": "निवडलेली प्रतिमा",
        "diagnose": "निदान करा",
        "analyzing": "प्रतिमेचे विश्लेषण करत आहे...",
        "disease_detected_en": "रोग शोधला गेला (English):",
        "disease_detected_lang": "रोग शोधला गेला",
        "suggestion_en": "सूचना (English)",
        "suggestion_lang": "सूचना",
        "audio_success": "ऑडिओ यशस्वीपणे तयार केले!",
        "audio_failed": "ऑडिओ तयार करण्यात अयशस्वी, परंतु मजकूर परिणाम वर उपलब्ध आहेत.",
        "translation_error": "भाषांतर उपलब्ध नाही. कृपया खालील इंग्रजी सूचना वाचा.",
        "non_plant_error": "ही प्रतिमा पानासारखी दिसत नाही. कृपया स्वच्छ पानाची प्रतिमा अपलोड करा.",
    },
}

# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Non_Plant",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ============================================================
# SUGGESTIONS DICTIONARY
# ============================================================

suggestions_dict = {
    # Non-Plant
    "Non_Plant": "This image does not appear to be a plant leaf. Please upload a clear image of a plant leaf for disease detection.",
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Practice crop rotation and tillage to reduce fungus residue. Consider resistant hybrids. Apply appropriate fungicides if disease is severe.",
    "Corn_(maize)___Common_rust_": "Plant resistant hybrids if available. Fungicide application is most effective when applied early, as rust spots first appear.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant corn hybrids. Crop rotation and tillage can help reduce disease. Fungicides can be effective if applied when lesions first appear.",
    "Corn_(maize)___healthy": "The plant appears healthy. Maintain good irrigation and nutrient management.",
    # Grape
    "Grape___Black_rot": "Remove and destroy infected vines and mummified grapes during dormancy. Improve air circulation through pruning. Apply protective fungicides.",
    "Grape___Esca_(Black_Measles)": "Prune out and destroy infected or dead wood during the dormant season. There is no chemical cure; management focuses on sanitation.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Often a late-season disease and may not require treatment. Improve air circulation. Rake and destroy fallen leaves.",
    "Grape___healthy": "The vine looks healthy. Continue with proper pruning, watering, and pest management.",
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": "This is a serious disease with no cure. Remove and destroy infected trees to prevent spread. Control the Asian Citrus Psyllid insect vector.",
    # Peach
    "Peach___Bacterial_spot": "Use resistant varieties. Apply copper-based bactericides during dormant and early growing season. Maintain good tree vigor.",
    "Peach___healthy": "The tree appears healthy. Maintain proper pruning, fertilization, and watering schedules.",
    # Pepper, Bell
    "Pepper,_bell___Bacterial_spot": "Avoid overhead watering. Plant disease-free seeds/transplants. Spray with copper-based bactericides in rotation.",
    "Pepper,_bell___healthy": "Excellent! The plant shows no signs of disease. Keep up the good work.",
    # Potato
    "Potato___Early_blight": "Remove affected lower leaves. Ensure good air circulation. Consider copper-based or chlorothalonil fungicide.",
    "Potato___Late_blight": "Serious disease. Remove infected plants immediately. Apply protective fungicides during cool, wet weather.",
    "Potato___healthy": "Your plant looks great! Continue with good watering and care practices.",
    # Soybean
    "Soybean___healthy": "The crop looks healthy. Continue monitoring for pests and diseases.",
    # Squash
    "Squash___Powdery_mildew": "Ensure good air circulation. Apply fungicides like sulfur, neem oil, or potassium bicarbonate at the first sign of disease.",
    # Strawberry
    "Strawberry___Leaf_scorch": "Remove infected leaves. Ensure proper spacing for air circulation. Water at the base to keep leaves dry.",
    "Strawberry___healthy": "The plant is healthy. Maintain consistent watering and protect from pests.",
    # Tomato
    "Tomato___Bacterial_spot": "Avoid overhead watering. Mulch around plants. Spray with copper-based bactericides.",
    "Tomato___Early_blight": "Prune lower leaves. Mulch to prevent soil splash. Ensure good air circulation.",
    "Tomato___Late_blight": "Serious disease. Remove infected plants. Apply fungicides preventatively.",
    "Tomato___Leaf_Mold": "Increase spacing and prune for better airflow. Reduce humidity if possible.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Water at base. Use mulch.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Pest issue. Spray insecticidal soap or neem oil on leaf undersides. Increase humidity to disrupt lifecycle.",
    "Tomato___Target_Spot": "Improve air circulation. Apply preventative fungicide. Water early in the day.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Viral disease. Remove infected plants. Control whiteflies with insecticides or sticky traps.",
    "Tomato___Tomato_mosaic_virus": "Viral disease. Remove infected plants. Wash hands and tools to prevent spread.",
    "Tomato___healthy": "The plant is healthy and strong. Continue your current care routine.",
    # Default
    "Default": "No specific suggestion available. Please consult a local agricultural expert.",
}

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image_for_disease(image: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNetB0 disease model."""
    img = image.convert("RGB").resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = efficientnet_preprocess(img_array)
    return img_array

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image: Image.Image):
    def predict_disease(image: Image.Image):
    if model is None:
        # Demo mode - return a sample prediction
        import random
        demo_classes = ["Tomato___healthy", "Potato___healthy", "Corn_(maize)___healthy"]
        predicted_class = random.choice(demo_classes)
        confidence = 0.85
        return predicted_class, confidence
    try:
        img_array = preprocess_image_for_disease(image)
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return "Unknown", 0.0

# ============================================================
# TRANSLATION & AUDIO HELPERS
# ============================================================

def translate_text(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def generate_audio(text: str, lang: str = "en") -> BytesIO:
    tts = gTTS(text=text, lang=lang)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# ============================================================
# SIDEBAR: NAVIGATION & SETTINGS
# ============================================================

with st.sidebar:
    # Logo / title
    st.markdown(
        """
        <div style='text-align: center; padding: 10px; 
                    background: linear-gradient(45deg, #4CAF50, #45a049); 
                    border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>गुरुप्रसाद कृषि सेवा केंद्र</h3>
            <p style='color: white; margin: 5px 0 0 0; font-size: 0.9em;'>
                Guruprasad Krushi Seva Kendra
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # UI language
    ui_lang_option = st.selectbox(
        "UI Language / यूआई भाषा:",
        options=["English", "मराठी"],
        index=0 if st.session_state.language == "english" else 1,
        key="ui_lang",
    )

    if ui_lang_option == "English":
        st.session_state.language = "english"
    else:
        st.session_state.language = "marathi"

    t = translations[st.session_state.language]

    st.markdown("---")

    # Navigation
    st.markdown(f"### {t['navigation']}")
    page_options = [t["home"], t["about"], t["contact"]]

    page = st.radio(
        t["select_page"],
        options=page_options,
        index=0
        if st.session_state.page == "Home"
        else (1 if st.session_state.page == "About Us" else 2),
        key="page_nav",
    )

    if page == t["home"]:
        st.session_state.page = "Home"
    elif page == t["about"]:
        st.session_state.page = "About Us"
    else:
        st.session_state.page = "Contact"

    st.markdown("---")

    # Detection language
    st.markdown(f"### {t['detection_language'].split(':')[0]}")
    detection_lang_option = st.selectbox(
        t["detection_language"],
        options=list(detection_languages.keys()),
        index=0,
        key="detection_lang",
    )
    target_lang = detection_languages[detection_lang_option]

    st.markdown("---")

    # Quick stats
    st.markdown(f"### {t['quick_stats']}")
    st.markdown(f"**{t['crops_supported']}**")
    st.markdown(f"**{t['languages_available']}**")
    st.markdown(f"**{t['ai_powered']}**")

# ============================================================
# MAIN PAGES
# ============================================================

# -------------------------- HOME ----------------------------
if st.session_state.page == "Home":
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    if st.session_state.language == "marathi":
        st.markdown(
            f"<h1 style='margin-top: 20px; font-size: 2.2rem; line-height: 1.3;'>{t['title']}</h1>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h1 style='margin-top: 20px;'>{t['title']}</h1>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<p class='subtext'>{t['subtitle']}</p>", unsafe_allow_html=True)
    st.divider()

    # Upload / camera
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["upload_gallery"])
        uploaded_file = st.file_uploader(
            t["choose_image"], type=["jpg", "jpeg", "png"]
        )
    with col2:
        st.subheader(t["capture_camera"])
        captured_image = st.camera_input(t["take_photo"])

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif captured_image is not None:
        image = Image.open(captured_image)

    if image:
        st.image(image, caption=t["selected_image"], use_container_width=True)

        if st.button(t["diagnose"]):
            with st.spinner(t["analyzing"]):
                # Disease prediction
                predicted_class, confidence = predict_disease(image)
                
                # Check if it's non-plant
                if predicted_class == "Non_Plant":
                    st.error(t["non_plant_error"])
                    st.stop()
                
                english_suggestion = suggestions_dict.get(
                    predicted_class, suggestions_dict["Default"]
                )

                # 3) Translations
                try:
                    translated_text = translate_text(english_suggestion, target_lang)
                except Exception:
                    translated_text = t["translation_error"]

                try:
                    translated_disease_text = translate_text(
                        f"Disease Detected: {predicted_class}", target_lang
                    )
                except Exception:
                    translated_disease_text = f"Disease Detected: {predicted_class}"

                # 4) Display results
                st.success(f"**{t['disease_detected_en']}** {predicted_class}")
                st.info(
                    f"**{t['disease_detected_lang']} ({detection_lang_option}):** "
                    f"{translated_disease_text}"
                )

                st.markdown(f"### {t['suggestion_en']}")
                st.write(english_suggestion)

                st.markdown(
                    f"### {t['suggestion_lang']} ({detection_lang_option})"
                )
                st.write(translated_text)

                # 5) Audio
                try:
                    audio_fp = generate_audio(translated_text, lang=target_lang)
                    st.audio(audio_fp, format="audio/mp3")
                    st.success(t["audio_success"])
                except Exception:
                    st.warning(t["audio_failed"])

# ------------------------- ABOUT US -------------------------
elif st.session_state.page == "About Us":
    if st.session_state.language == "english":
        st.markdown("# Guruprasad Krushi Seva Kendra")
        st.markdown(
            """
        **Guruprasad Krushi Seva Kendra** is an AI-powered agriculture support system that helps farmers detect crop diseases early using image recognition.  
        It provides instant, multilingual plant health diagnosis and simple treatment suggestions to help farmers make better decisions.

        **Mission:** Helping farmers detect plant diseases early and get effective treatments.  
        **Vision:** To empower rural farmers with AI-based plant health support.

        ### Crops Supported for Diagnosis:
        - Corn (Maize)  
        - Grape  
        - Orange  
        - Peach  
        - Pepper (Bell)  
        - Potato  
        - Soybean  
        - Squash  
        - Strawberry  
        - Tomato  

        ### How It Works:
        1. Upload or capture an image of a plant leaf  
        2. AI analyzes the image to detect crop type and disease  
        3. Get instant diagnosis with treatment suggestions  
        4. Audio support in 10 languages  
        5. Switch between English and Marathi interface  

        ### Contact Information
        **Address:** Dhotrewadi, Sangli  
        **Phone:** +91 7620450915
        
        ### Features
        - **Instant Detection** - Get results in seconds  
        - **Multi-language Support** - 10 languages available  
        - **Mobile Friendly** - Works on all devices  
        - **Expert Suggestions** - Farmer-friendly advice  
        """
        )
    else:
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='font-size: 1.8rem; line-height: 1.4; text-align: center; "
            "color: #1b4332; padding: 10px; margin-bottom: 20px; word-spacing: 2px;'>"
            "गुरुकृपा कृषि सेवा केंद्र</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        **गुरुप्रसाद कृषि सेवा केंद्र** हे एक AI-आधारित कृषी सहाय्य प्रणाली आहे जी शेतकऱ्यांना प्रतिमा ओळख वापरून पिकांच्या रोगांची लवकर ओळख करण्यात मदत करते.  
        हे शेतकऱ्यांना चांगले निर्णय घेण्यास मदत करण्यासाठी तत्काळ, बहुभाषिक वनस्पती आरोग्य निदान आणि सोपे उपचार सूचना प्रदान करते.

        **ध्येय:** शेतकऱ्यांना वनस्पतींच्या रोगांची लवकर ओळख करण्यात आणि प्रभावी उपचार मिळविण्यात मदत करणे.  
        **दृष्टिकोन:** AI-आधारित वनस्पती आरोग्य सहाय्याने ग्रामीण शेतकऱ्यांना सशक्त करणे.

        ### निदानासाठी समर्थित पिके:
        - मका  
        - द्राक्ष  
        - संत्रा  
        - पीच  
        - मिरची (बेल)  
        - बटाटा  
        - सोयाबीन  
        - भोपळा  
        - स्ट्रॉबेरी  
        - टोमॅटो  

        ### हे कसे कार्य करते:
        1. वनस्पतीच्या पानाची प्रतिमा अपलोड करा किंवा कॅप्चर करा  
        2. AI पिकाचा प्रकार आणि रोग शोधण्यासाठी प्रतिमेचे विश्लेषण करते  
        3. उपचार सूचनांसह तत्काळ निदान मिळवा  
        4. 10 भाषांमध्ये ऑडिओ समर्थन  
        5. इंग्रजी आणि मराठी इंटरफेसमध्ये स्विच करा  

        ### संपर्क माहिती
        **पत्ता:** धोत्रेवाडी, सांगली  
        **फोन:** +91 7620450915
        
        ### वैशिष्ट्ये
        - **तत्काळ शोध** - सेकंदांत परिणाम  
        - **बहुभाषिक समर्थन** - 10 भाषा उपलब्ध  
        - **मोबाइल-अनुकूल** - सर्व उपकरणांवर कार्य करते  
        - **तज्ञ सूचना** - शेतकरी-अनुकूल सल्ला  
        """
        )

# ------------------------- CONTACT -------------------------
elif st.session_state.page == "Contact":
    if st.session_state.language == "english":
        st.markdown("# Contact Guruprasad Krushi Seva Kendra")
        st.markdown(
            """
        ### Office Information
        **Name:** Guruprasad Krushi Seva Kendra  
        **Address:** Dhotrewadi, Sangli, Maharashtra, India  
        **Phone:** +91 7620450915
        
        ### Working Hours
        **Monday to Saturday:** 9:00 AM - 6:00 PM  
        **Sunday:** 10:00 AM - 4:00 PM
        
        ### Services Offered
        - AI-powered plant disease detection  
        - Multi-language support (10 languages)  
        - Expert agricultural advice  
        - Mobile-friendly platform  
        - Audio guidance for farmers
        
        ### Get in Touch
        For any queries, technical support, or feedback, please contact us using the information above.  
        We are committed to helping farmers with the best agricultural technology solutions.
        """
        )
    else:
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='font-size: 1.8rem; line-height: 1.4; text-align: center; "
            "color: #1b4332; padding: 10px; margin-bottom: 20px; word-spacing: 2px;'>"
            "गुरुकृपा कृषि सेवा केंद्र संपर्क</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        ### कार्यालय माहिती
        **नाव:** गुरुप्रसाद कृषि सेवा केंद्र  
        **पत्ता:** धोत्रेवाडी, सांगली, महाराष्ट्र, भारत  
        **फोन:** +91 7620450915
        
        ### कार्य वेळा
        **सोमवार ते शनिवार:** सकाळी 9:00 - संध्याकाळी 6:00  
        **रविवार:** सकाळी 10:00 - दुपारी 4:00
        
        ### सेवा
        - AI-आधारित वनस्पती रोग शोध  
        - बहुभाषिक समर्थन (10 भाषा)  
        - तज्ञ कृषी सल्ला  
        - मोबाइल-अनुकूल प्लॅटफॉर्म  
        - शेतकऱ्यांसाठी ऑडिओ मार्गदर्शन  
        
        ### संपर्कात रहा
        कोणत्याही प्रश्न, तांत्रिक सहाय्य किंवा अभिप्रायासाठी, कृपया वरील माहिती वापरून आमच्याशी संपर्क साधा.  
        आम्ही शेतकऱ्यांना सर्वोत्तम कृषी तंत्रज्ञान समाधान देण्यासाठी वचनबद्ध आहोत.
        """
        )
