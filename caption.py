import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_captioner import ArabicImageCaptioner

# Paths
VISION_MODEL = '../finetuning-system/models/simple_vit_arabic_food'
TEST_DIR = '../finetuning-system/data/arabic_food/test'

print("\n🎨 ARABIC IMAGE CAPTIONING\n")

# Initialize
captioner = ArabicImageCaptioner(vision_model_path=VISION_MODEL)

# Test on all classes
test_path = Path(TEST_DIR)

print("\n" + "="*80)
print("GENERATING CAPTIONS FOR ALL FOOD TYPES")
print("="*80)

for class_dir in sorted(test_path.iterdir()):
    if class_dir.is_dir():
        # Get first image
        images = list(class_dir.glob('*.jpg'))
        if images:
            img = images[0]
            
            # Generate caption
            result = captioner.generate_caption(str(img))
            
            print(f"\n📸 {class_dir.name}")
            print(f"   {result['caption']}")
            print(f"   (Confidence: {result['confidence']*100:.1f}%)")

print("\n" + "="*80)
print("✅ Captioning complete!")
print("="*80 + "\n")