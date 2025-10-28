import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
from googletrans import Translator
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Smart Plant Disease Diagnosis System",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fa; }
    h1 { text-align: center; color: #1b4332; margin-bottom: 0; }
    .subtext { text-align: center; color: #4b5563; margin-top: 5px; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("EfficientNet_plant_disease_model.keras")

model = load_model()
translator = Translator()

# ============================================================
# LANGUAGE OPTIONS
# ============================================================
languages = {
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
    "Urdu": "ur"
}

# ============================================================
# CLASS NAMES
# ============================================================
class_names = [
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

# ============================================================
# SUGGESTIONS DICTIONARY
# ============================================================
suggestions_dict = { 
    # Corn (Maize)
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Practice crop rotation and tillage to reduce fungus residue. Consider resistant hybrids. Apply appropriate fungicides if disease is severe.",
    'Corn_(maize)___Common_rust_': "Plant resistant hybrids if available. Fungicide application is most effective when applied early, as rust spots first appear.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant corn hybrids. Crop rotation and tillage can help reduce disease. Fungicides can be effective if applied when lesions first appear.",
    'Corn_(maize)___healthy': "The plant appears healthy. Maintain good irrigation and nutrient management.",

    # Grape
    'Grape___Black_rot': "Remove and destroy infected vines and mummified grapes during dormancy. Improve air circulation through pruning. Apply protective fungicides.",
    'Grape___Esca_(Black_Measles)': "Prune out and destroy infected or dead wood during the dormant season. There is no chemical cure; management focuses on sanitation.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Often a late-season disease and may not require treatment. Improve air circulation. Rake and destroy fallen leaves.",
    'Grape___healthy': "The vine looks healthy. Continue with proper pruning, watering, and pest management.",

    # Orange
    'Orange___Haunglongbing_(Citrus_greening)': "This is a serious disease with no cure. Remove and destroy infected trees to prevent spread. Control the Asian Citrus Psyllid insect vector.",

    # Peach
    'Peach___Bacterial_spot': "Use resistant varieties. Apply copper-based bactericides during dormant and early growing season. Maintain good tree vigor.",
    'Peach___healthy': "The tree appears healthy. Maintain proper pruning, fertilization, and watering schedules.",

    # Pepper, Bell
    'Pepper,_bell___Bacterial_spot': "Avoid overhead watering. Plant disease-free seeds/transplants. Spray with copper-based bactericides in rotation.",
    'Pepper,_bell___healthy': "Excellent! The plant shows no signs of disease. Keep up the good work.",

    # Potato
    'Potato___Early_blight': "Remove affected lower leaves. Ensure good air circulation. Consider copper-based or chlorothalonil fungicide.",
    'Potato___Late_blight': "Serious disease. Remove infected plants immediately. Apply protective fungicides during cool, wet weather.",
    'Potato___healthy': "Your plant looks great! Continue with good watering and care practices.",

    # Soybean
    'Soybean___healthy': "The crop looks healthy. Continue monitoring for pests and diseases.",

    # Squash
    'Squash___Powdery_mildew': "Ensure good air circulation. Apply fungicides like sulfur, neem oil, or potassium bicarbonate at the first sign of disease.",

    # Strawberry
    'Strawberry___Leaf_scorch': "Remove infected leaves. Ensure proper spacing for air circulation. Water at the base to keep leaves dry.",
    'Strawberry___healthy': "The plant is healthy. Maintain consistent watering and protect from pests.",

    # Tomato
    'Tomato___Bacterial_spot': "Avoid overhead watering. Mulch around plants. Spray with copper-based bactericides.",
    'Tomato___Early_blight': "Prune lower leaves. Mulch to prevent soil splash. Ensure good air circulation.",
    'Tomato___Late_blight': "Serious disease. Remove infected plants. Apply fungicides preventatively.",
    'Tomato___Leaf_Mold': "Increase spacing and prune for better airflow. Reduce humidity if possible.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves. Water at base. Use mulch.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Pest issue. Spray insecticidal soap or neem oil on leaf undersides. Increase humidity to disrupt lifecycle.",
    'Tomato___Target_Spot': "Improve air circulation. Apply preventative fungicide. Water early in the day.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Viral disease. Remove infected plants. Control whiteflies with insecticides or sticky traps.",
    'Tomato___Tomato_mosaic_virus': "Viral disease. Remove infected plants. Wash hands and tools to prevent spread.",
    'Tomato___healthy': "The plant is healthy and strong. Continue your current care routine.",
    
    # Default fallback
    'Default': "No specific suggestion available. Please consult a local agricultural expert."
}

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image):
    img = image.convert("RGB").resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing
    return img_array

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_disease(image):
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return "Unknown", 0.0

# ============================================================
# AUDIO GENERATION
# ============================================================
def generate_audio(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# ============================================================
# MAIN PAGE CONTENT
# ============================================================
st.markdown("<h1>Smart Plant Disease Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload or capture a plant image to detect disease and receive treatment suggestions.</p>", unsafe_allow_html=True)
st.divider()

# Language selection
selected_language = st.selectbox(
    "Select your preferred language for translation & audio alert:",
    options=list(languages.keys()),
    index=0
)
target_lang = languages[selected_language]

# Upload & capture section
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload from Gallery")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
with col2:
    st.subheader("Capture from Camera")
    captured_image = st.camera_input("Take a photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif captured_image is not None:
    image = Image.open(captured_image)

if image:
    st.image(image, caption="Selected Image", use_container_width=True)
    if st.button("Diagnose"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence = predict_disease(image)
            english_suggestion = suggestions_dict.get(predicted_class, suggestions_dict["Default"])
            
            try:
                translated_text = translator.translate(english_suggestion, src="en", dest=target_lang).text
            except:
                translated_text = "Translation unavailable. Please read English suggestion below."

            # st.success(f"**Disease Detected:** {predicted_class}")
            # #st.info(f"**Confidence:** {confidence * 100:.2f}%")
            # st.markdown("### Suggestion (English)")
            # st.write(english_suggestion)
            # st.markdown(f"### Suggestion ({selected_language})")
            # st.write(translated_text)
            # Translate disease name and label into selected language
            # Translate disease name and label into selected language
            try:
                translated_disease_text = translator.translate(
                    f"Disease Detected: {predicted_class}", src="en", dest=target_lang
                ).text
            except:
                translated_disease_text = f"Disease Detected: {predicted_class}"

            # =========================
            # DISPLAY RESULTS
            # =========================
            st.success(f"**Disease Detected (English):** {predicted_class}")
            st.info(f"**Disease Detected ({selected_language}):** {translated_disease_text}")

            # Optional: show model confidence
            # st.caption(f"Confidence: {confidence * 100:.2f}%")

            st.markdown("### Suggestion (English)")
            st.write(english_suggestion)

            st.markdown(f"### Suggestion ({selected_language})")
            st.write(translated_text)



            audio_fp = generate_audio(translated_text, lang=target_lang)
            st.audio(audio_fp, format="audio/mp3")
