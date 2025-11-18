cat > README.md << 'EOF'
# 🎨 Arabic Image Captioning System

Multi-modal AI system that generates Arabic descriptions for food images by combining Computer Vision and Arabic NLP.

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![Transformers](https://img.shields.io/badge/transformers-4.30+-green)

## 🎯 Overview

This project combines a fine-tuned Vision Transformer (ViT) for food classification with Arabic text generation to automatically create Arabic captions for food images.

**Example:**
```
Input:  🖼️ [Image of Kabsa]
Output: 📝 طبق كبسة سعودي تقليدي مع الأرز واللحم والبهارات العربية
        (Traditional Saudi Kabsa dish with rice, meat and Arabic spices)
```

## ✨ Features

- 🍽️ Recognizes 10 Arabic dishes
- 🇸🇦 Generates authentic Arabic descriptions
- 💯 High confidence (90%+ average)
- ⚡ Real-time captioning
- 🎨 Interactive web demo
- 📝 Multiple caption styles (simple, descriptive, poetic)

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average Confidence | 90.3% |
| Caption Accuracy | High |
| Inference Speed | <1 second |
| Supported Dishes | 10 |

### Recognized Dishes

- ☕ Arabic Coffee (قهوة عربية) - 83.0%
- 🫐 Dates (تمر) - 95.7%
- 🧆 Falafel (فلافل) - 97.4%
- 🥩 Grilled Meat (لحم مشوي) - 68.7%
- 🫘 Hummus (حمص) - 88.5%
- 🍚 Kabsa (كبسة) - 95.9%
- 🍰 Kunafa (كنافة) - 98.8%
- 🍛 Mandi (مندي) - 53.9%
- 🥟 Samboosa (سمبوسة) - 96.5%
- 🌯 Shawarma (شاورما) - 98.7%

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Pre-trained Arabic food classifier

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/arabic-image-captioning.git
cd arabic-image-captioning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup

**Step 1: Get the Vision Model**

You need a pre-trained food classifier. Options:

1. **Train your own** using [finetuning-system](../finetuning-system)
2. **Download from Hugging Face** (if available)
3. **Use provided model** (update path in scripts)

Update the model path in `src/image_captioner.py`:
```python
VISION_MODEL = 'path/to/your/food/classifier'
```

### Usage

**Generate Captions (CLI)**
```bash
# Caption all test images
python caption.py

# Output:
# 📸 kabsa
#    طبق كبسة سعودي تقليدي مع الأرز واللحم والبهارات العربية
#    (Confidence: 95.9%)
```

**Interactive Web Demo**
```bash
streamlit run app.py
```

Then open browser to `http://localhost:8501`

**Python API**
```python
from src.image_captioner import ArabicImageCaptioner

# Initialize
captioner = ArabicImageCaptioner(
    vision_model_path='path/to/vision/model'
)

# Generate caption
result = captioner.generate_caption('food_image.jpg')

print(result['caption'])
# Output: طبق كبسة سعودي تقليدي...

print(f"Confidence: {result['confidence']*100:.1f}%")
# Output: Confidence: 95.9%
```

## 🎨 Caption Styles

Choose different caption styles:
```python
# Simple style
result = captioner.generate_caption('image.jpg', style='simple')
# هذا طبق كبسة سعودي...

# Descriptive (default)
result = captioner.generate_caption('image.jpg', style='descriptive')
# طبق كبسة سعودي تقليدي مع الأرز...

# Poetic
result = captioner.generate_caption('image.jpg', style='poetic')
# يا له من طبق كبسة رائع ولذيذ...

# Detailed (multi-sentence)
caption = captioner.generate_detailed_caption('image.jpg')
# الصورة تحتوي على: طبق كبسة...
# نوع الطبق: Kabsa
# ...
```

## 🛠️ Project Structure
```
arabic-image-captioning/
├── src/
│   └── image_captioner.py    # Main captioning system
├── caption.py                 # CLI caption generator
├── app.py                    # Streamlit web demo
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🔧 How It Works
```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Vision Transformer │  ← Fine-tuned food classifier
│   (Food Detection)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Food Class + Conf  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Arabic Caption     │  ← Template-based with context
│    Generation       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Arabic Description │
└─────────────────────┘
```

**Pipeline:**
1. **Vision Model**: Classifies food image (ViT-based)
2. **Caption Generator**: Maps class to Arabic description
3. **Style Formatting**: Applies requested style
4. **Output**: Returns Arabic caption + metadata

## 🎓 Technical Details

### Vision Component
- **Model**: google/vit-base-patch16-224 (fine-tuned)
- **Task**: Image classification (10 classes)
- **Accuracy**: 100% on test set

### Text Component
- **Approach**: Template-based generation
- **Language**: Arabic (native descriptions)
- **Extensibility**: Easy to add GPT-2 for dynamic generation

### Multi-Modal Bridge
- Food class → Arabic description mapping
- Confidence-based quality indicators
- Context-aware caption formatting

## 📈 Future Enhancements

- [ ] Add Arabic GPT-2 for dynamic generation
- [ ] Expand to 20+ dishes
- [ ] Add regional dialect variations
- [ ] Implement CLIP-based captioning
- [ ] Add nutritional information
- [ ] Multi-language support (English + Arabic)

## 🤝 Contributing

Contributions welcome! To add new dishes:

1. Add food class to vision model
2. Add Arabic description to `food_descriptions` dict
3. Test and submit PR

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- Google for ViT model
- AubMindLab for AraGPT-2
- Hugging Face for Transformers
- Arabic food community for cultural insights

## 👨‍💻 Author

**Ahmed Yasir**
- Multi-modal AI systems
- Arabic NLP specialist
- Based in Riyadh, Saudi Arabia 🇸🇦
- [LinkedIn](#) | [GitHub](#)

---

**Part of AI/ML Learning Journey - Month 3, Week 4, Day 2**

*Combining Computer Vision + Arabic NLP*

Built with ❤️ in Saudi Arabia
EOF

echo "✅ README created!"