import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from diffusers import StableDiffusionInpaintingPipeline
import torch
import os
import gc

# Global model initialization - MUST be outside handler
print("Starting model initialization...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize model with error handling
try:
    pipe = StableDiffusionInpaintingPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        # Enable CPU offload to prevent memory issues
        pipe.enable_model_cpu_offload()
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

def detect_black_frame(image):
    """Simple but effective black frame detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check edges for black pixels
    edge_thickness = int(min(w, h) * 0.15)  # Check 15% from each edge
    
    # Define edge regions
    top = gray[:edge_thickness, :]
    bottom = gray[-edge_thickness:, :]
    left = gray[:, :edge_thickness]
    right = gray[:, -edge_thickness:]
    
    # Calculate mean brightness
    threshold = 30  # Adjusted for better detection
    
    has_black_frame = (
        np.mean(top) < threshold or
        np.mean(bottom) < threshold or
        np.mean(left) < threshold or
        np.mean(right) < threshold
    )
    
    return has_black_frame

def remove_black_frame(image):
    """Remove black frame by cropping"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find non-black pixels
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        # Resize back to original size
        result = cv2.resize(cropped, (image.shape[1], image.shape[0]), 
                           interpolation=cv2.INTER_LANCZOS4)
        return result
    
    return image

def create_ring_mask(image_size):
    """Create center mask for ring enhancement"""
    h, w = image_size[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create center circle mask
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3
    
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Blur edges for smooth transition
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

def enhance_with_inpainting(image, mask):
    """Enhance ring area using Stable Diffusion Inpainting"""
    if pipe is None:
        print("Model not loaded, skipping inpainting")
        return image
    
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        
        # Resize for processing
        process_size = (512, 512)
        original_size = pil_image.size
        
        pil_image_resized = pil_image.resize(process_size, Image.Resampling.LANCZOS)
        pil_mask_resized = pil_mask.resize(process_size, Image.Resampling.LANCZOS)
        
        # Generate enhanced image
        with torch.no_grad():
            result = pipe(
                prompt="ultra high quality luxury wedding ring, professional jewelry photography, sharp details, perfect lighting",
                negative_prompt="blurry, low quality, distorted, deformed",
                image=pil_image_resized,
                mask_image=pil_mask_resized,
                num_inference_steps=30,
                guidance_scale=7.5,
                strength=0.8
            ).images[0]
        
        # Resize back
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Convert back to numpy
        enhanced = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return enhanced
        
    except Exception as e:
        print(f"Inpainting error: {e}")
        return image

def apply_post_processing(image):
    """Apply final enhancements"""
    # Convert to PIL for processing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.3)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Enhance color
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.05)
    
    # Slight unsharp mask
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
    
    # Convert back
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result

def handler(event):
    """RunPod handler function"""
    try:
        print("Handler started")
        
        # Get input
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        
        if not image_base64:
            return {"error": "No image provided"}
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image"}
        
        print(f"Processing image: {image.shape}")
        
        # Step 1: Check and remove black frame
        if detect_black_frame(image):
            print("Black frame detected, removing...")
            image = remove_black_frame(image)
        
        # Step 2: Create ring mask
        mask = create_ring_mask(image.shape)
        
        # Step 3: Enhance with inpainting
        enhanced = enhance_with_inpainting(image, mask)
        
        # Step 4: Apply post-processing
        final_result = apply_post_processing(enhanced)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', final_result, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print("Processing completed successfully")
        
        return {
            "enhanced_image": result_base64,
            "status": "success",
            "message": "Image processed successfully"
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "status": "failed"
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
