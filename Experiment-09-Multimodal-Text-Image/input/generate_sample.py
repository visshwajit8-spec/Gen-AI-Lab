"""Helper script to create sample image for multimodal experiment."""
from pathlib import Path
from PIL import Image, ImageDraw

def create_dog_image(target_path: Path):
    img = Image.new("RGB", (512, 384), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Grass field
    draw.rectangle([0, 200, 512, 384], fill=(76, 175, 80))
    for y in range(200, 384, 15):
        for x in range(0, 512, 20):
            draw.line([(x, y), (x + 4, y - 10)], fill=(56, 142, 60), width=2)
    # Dog body (brown/golden retriever)
    draw.ellipse([180, 180, 310, 260], fill=(184, 115, 51))
    draw.ellipse([270, 140, 340, 210], fill=(184, 115, 51))
    draw.ellipse([320, 155, 345, 175], fill=(50, 30, 20)) # muzzle
    draw.ellipse([300, 155, 310, 165], fill=(0, 0, 0)) # eye
    # Dog legs
    draw.line([(200, 240), (195, 300)], fill=(184, 115, 51), width=10)
    draw.line([(230, 245), (235, 300)], fill=(184, 115, 51), width=10)
    draw.line([(270, 245), (265, 300)], fill=(184, 115, 51), width=10)
    draw.line([(295, 240), (300, 300)], fill=(184, 115, 51), width=10)
    # Dog tail
    draw.arc([140, 160, 200, 230], start=180, end=300, fill=(184, 115, 51), width=10)
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path)

if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "sample_image.jpg"
    create_dog_image(out_path)
