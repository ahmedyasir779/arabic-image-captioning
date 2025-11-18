import torch
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from PIL import Image
import json

class ArabicImageCaptioner:
    """
    Generate Arabic captions for food images
    """
    
    def __init__(
        self,
        vision_model_path: str,
        text_model_path: str = 'aubmindlab/aragpt2-base'
    ):
        """
        Initialize captioner
        
        Args:
            vision_model_path: Path to fine-tuned food classifier
            text_model_path: Path to Arabic text generation model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Initializing Arabic Image Captioner...")
        print(f"   Device: {self.device}")
        
        # Load vision model
        print(f"\n📸 Loading vision model...")
        self.vision_model = AutoModelForImageClassification.from_pretrained(
            vision_model_path
        )
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_path)
        self.vision_model.to(self.device)
        self.vision_model.eval()
        
        # Load class mapping
        with open(f'{vision_model_path}/class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        self.idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
        
        print(f"   ✅ Vision model loaded")
        print(f"   Classes: {len(self.idx_to_class)}")
        
        # Load text generation model
        print(f"\n📝 Loading text generation model...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model_path)
        
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        self.text_model.to(self.device)
        self.text_model.eval()
        
        print(f"   ✅ Text model loaded")
        
        # Arabic food descriptions (template-based)
        self.food_descriptions = {
            'kabsa': 'طبق كبسة سعودي تقليدي مع الأرز واللحم والبهارات العربية',
            'mandi': 'طبق مندي يمني شهير يحتوي على أرز ولحم مطبوخ بطريقة خاصة',
            'shawarma': 'شاورما لذيذة مع اللحم المتبل والخضروات الطازجة',
            'kunafa': 'حلوى كنافة عربية تقليدية مع الجبن والقطر الحلو',
            'falafel': 'فلافل مقرمشة من الحمص والبقوليات مع التوابل',
            'hummus': 'حمص ناعم مع زيت الزيتون والطحينة',
            'dates': 'تمر طازج وحلو من أجود أنواع التمور',
            'arabic_coffee': 'قهوة عربية أصيلة بنكهة الهيل والزعفران',
            'samboosa': 'سمبوسة مقرمشة محشوة باللحم أو الخضار',
            'grilled_meat': 'لحم مشوي طازج متبل بالبهارات العربية'
        }
        
        print(f"\n✅ Captioner ready!")
    
    def classify_image(self, image_path: str):
        """
        Classify food image
        
        Args:
            image_path: Path to image
            
        Returns:
            dict with class and confidence
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process
        inputs = self.image_processor(image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        food_class = self.idx_to_class[pred_idx]
        
        return {
            'class': food_class,
            'confidence': confidence,
            'probabilities': {
                self.idx_to_class[i]: probs[i].item()
                for i in range(len(self.idx_to_class))
            }
        }
    
    def generate_caption(self, image_path: str, style: str = 'descriptive'):
        """
        Generate Arabic caption for image
        
        Args:
            image_path: Path to image
            style: 'descriptive', 'poetic', or 'simple'
            
        Returns:
            dict with caption and metadata
        """
        # Classify image
        classification = self.classify_image(image_path)
        food_class = classification['class']
        confidence = classification['confidence']
        
        # Get base description
        base_description = self.food_descriptions.get(
            food_class,
            f"طبق {food_class.replace('_', ' ')}"
        )
        
        # Style variations
        if style == 'simple':
            caption = f"هذا {base_description}"
        elif style == 'poetic':
            caption = f"يا له من {base_description} رائع ولذيذ"
        else:  # descriptive
            caption = base_description
        
        # Add confidence-based commentary
        if confidence > 0.9:
            quality = "واضح جداً"
        elif confidence > 0.7:
            quality = "جيد"
        else:
            quality = "محتمل"
        
        return {
            'caption': caption,
            'food_class': food_class,
            'confidence': confidence,
            'confidence_text': quality,
            'english_name': food_class.replace('_', ' ').title(),
            'all_probabilities': classification['probabilities']
        }
    
    def generate_detailed_caption(self, image_path: str):
        """
        Generate detailed multi-sentence caption
        
        Args:
            image_path: Path to image
            
        Returns:
            Detailed Arabic caption
        """
        result = self.generate_caption(image_path)
        
        caption = f"""
الصورة تحتوي على: {result['caption']}

نوع الطبق: {result['english_name']}
مستوى الثقة: {result['confidence']*100:.1f}%

{result['caption']} - طبق تقليدي من المطبخ العربي يتميز بنكهته المميزة ومكوناته الطازجة.
        """.strip()
        
        return caption


def main():
    """Test captioner"""
    print("\n🎨 ARABIC IMAGE CAPTIONING DEMO\n")
    
    # Paths (update these!)
    VISION_MODEL = '../finetuning-system/models/simple_vit_arabic_food'
    
    # Initialize
    captioner = ArabicImageCaptioner(vision_model_path=VISION_MODEL)
    
    # Test image
    test_image = '../finetuning-system/data/arabic_food/test/kabsa'
    
    from pathlib import Path
    test_images = list(Path(test_image).glob('*.jpg'))
    
    if test_images:
        test_img = str(test_images[0])
        
        print("\n" + "="*70)
        print("GENERATING CAPTION")
        print("="*70)
        
        # Generate caption
        result = captioner.generate_caption(test_img)
        
        print(f"\n📸 Image: {Path(test_img).name}")
        print(f"\n📝 Caption:")
        print(f"   {result['caption']}")
        print(f"\n🍽️  Food: {result['english_name']}")
        print(f"💯 Confidence: {result['confidence']*100:.1f}%")
        
        # Detailed caption
        print("\n" + "="*70)
        print("DETAILED CAPTION")
        print("="*70)
        
        detailed = captioner.generate_detailed_caption(test_img)
        print(f"\n{detailed}")
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()