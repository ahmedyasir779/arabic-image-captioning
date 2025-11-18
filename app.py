import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_captioner import ArabicImageCaptioner
from PIL import Image
import io

st.set_page_config(
    page_title="Arabic Image Captioning",
    page_icon="🎨",
    layout="centered"
)

# Title
st.title("🎨 مولد التعليقات العربية للصور")
st.title("Arabic Image Caption Generator")
st.markdown("**Upload a food image and get an Arabic description!**")
st.markdown("---")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This system combines Computer Vision and Arabic NLP "
    "to automatically generate Arabic descriptions for food images."
)

st.sidebar.header("How it works")
st.sidebar.markdown("""
1. 🖼️ Upload a food image
2. 🤖 AI identifies the dish
3. 📝 Generates Arabic caption
4. ✨ Shows detailed description
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Ahmed Yasir**")
st.sidebar.markdown("🇸🇦 Made in Saudi Arabia")

# Load model
@st.cache_resource
def load_captioner():
    VISION_MODEL = '../finetuning-system/models/simple_vit_arabic_food'
    return ArabicImageCaptioner(vision_model_path=VISION_MODEL)

with st.spinner("Loading models..."):
    captioner = load_captioner()

st.success("✅ Models loaded!")

# File uploader
uploaded_file = st.file_uploader(
    "📸 Upload a food image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        # Save temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Generate caption
        with st.spinner("🔮 Generating caption..."):
            result = captioner.generate_caption(temp_path, style='descriptive')
            detailed = captioner.generate_detailed_caption(temp_path)
        
        st.subheader("📝 التعليق")
        st.success(result['caption'])
        
        st.metric("🍽️ Dish", result['english_name'])
        st.metric("💯 Confidence", f"{result['confidence']*100:.1f}%")
    
    # Detailed caption
    st.markdown("---")
    st.subheader("📄 Detailed Description")
    st.text_area("", detailed, height=200)
    
    # Top predictions
    st.markdown("---")
    st.subheader("📊 All Predictions")
    
    probs = result['all_probabilities']
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    for food, prob in sorted_probs[:5]:
        st.progress(prob)
        st.write(f"{food.replace('_', ' ').title()}: {prob*100:.1f}%")

else:
    st.info("👆 Upload an image to get started!")
    
    # Example
    st.markdown("---")
    st.subheader("💡 Example Output")
    st.markdown("""
    **Input:** Image of Kabsa  
    **Output:** طبق كبسة سعودي تقليدي مع الأرز واللحم والبهارات العربية  
    **Translation:** Traditional Saudi Kabsa dish with rice, meat and Arabic spices
    """)

# Footer
st.markdown("---")
st.markdown(
    "🚀 **Multi-Modal AI** | "
    "🎯 **Computer Vision + Arabic NLP** | "
    "⚡ **Real-time captioning**"
)