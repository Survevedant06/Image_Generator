import os
import sys
import glob
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image, ImageOps

# Enable HEIC / HEIF format support for smartphone photos
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"
)


def load_birefnet_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BiRefNet on {device}...")
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float32,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        **model_kwargs
    )
    model.to(device)
    model.float()
    model.eval()
    return model, device


def load_and_standardize_image(image_path: str) -> Image.Image:
    """Safely opens any image format, normalizes orientation, and returns standard RGB."""
    raw_img = Image.open(image_path)
    raw_img = ImageOps.exif_transpose(raw_img)

    if raw_img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", raw_img.size, (255, 255, 255))
        alpha_channel = raw_img.split()[-1]
        background.paste(raw_img, mask=alpha_channel)
        return background
    elif raw_img.mode != "RGB":
        return raw_img.convert("RGB")
    
    return raw_img


def remove_background(image_path: str, output_path: str, model, device: str):
    raw_image = load_and_standardize_image(image_path)
    original_size = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(raw_image).unsqueeze(0).to(device).float()

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    pred_mask = preds[0].squeeze().float()
    mask_pil = transforms.ToPILImage()(pred_mask)
    mask_pil = mask_pil.resize(original_size, Image.Resampling.BILINEAR)

    result_image = raw_image.copy()
    result_image.putalpha(mask_pil)

    # Output is always saved as PNG with alpha channel
    base_name = os.path.splitext(output_path)[0]
    final_output_path = f"{base_name}.png"
    result_image.save(final_output_path, format="PNG")

    # Also save inside processed_images/ for pipeline state tracking
    processed_dir = "processed_images"
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"{os.path.splitext(os.path.basename(final_output_path))[0]}.png")
    result_image.save(processed_path, format="PNG")
    print(f"Done! Cutout saved to: {final_output_path} and {processed_path}")


def find_any_local_image():
    """Detects any image file present in the current folder."""
    for file in os.listdir("."):
        if file.lower().endswith(SUPPORTED_EXTENSIONS) and not file.endswith("_cutout.png"):
            return file
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # File provided as argument
        target_file = sys.argv[1]
        if not os.path.exists(target_file):
            print(f"[Error] File not found: {target_file}")
            sys.exit(1)
        print(f"Selected input image: {target_file}")
        output_name = f"{os.path.splitext(target_file)[0]}_cutout.png"
        model, device = load_birefnet_model()
        remove_background(target_file, output_name, model, device)
    else:
        # No argument passed: seamlessly run smart batch background removal
        from batch_remove_bg import process_batch
        process_batch()