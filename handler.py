#!/usr/bin/env python3
"""
Wedding Ring AI v107 - Clean Architecture
- No black border detection needed
- Direct wedding ring detection using Replicate
- Fixed 1000x1300 output size
- Simplified and optimized workflow
"""

import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
from io import BytesIO
import traceback
import os
import replicate
from typing import Dict, Tuple, Optional

# Wedding ring enhancement parameters (28 pairs of training data)
WEDDING_RING_PARAMS = {
    "white_gold": {
        "brightness": 1.18,
        "contrast": 1.12,
        "sharpness": 1.85,
        "saturation": 0.82,
        "white_overlay": 0.12,
        "gamma": 0.93,
        "highlights": 1.15,
        "shadows": 0.85,
        "temperature": -5,
        "vibrance": 1.1
    },
    "rose_gold": {
        "brightness": 1.2,
        "contrast": 1.15,
        "sharpness": 1.88,
        "saturation": 0.9,
        "white_overlay": 0.08,
        "gamma": 0.92,
        "highlights": 1.12,
        "shadows": 0.88,
        "temperature": -3,
        "vibrance": 1.15
    },
    "yellow_gold": {
        "brightness": 1.25,
        "contrast": 1.2,
        "sharpness": 1.92,
        "saturation": 0.95,
        "white_overlay": 0.05,
        "gamma": 0.88,
        "highlights": 1.18,
        "shadows": 0.82,
        "temperature": -2,
        "vibrance": 1.2
    },
    "white": {
        "brightness": 1.15,
        "contrast": 1.08,
        "sharpness": 1.8,
        "saturation": 0.88,
        "white_overlay": 0.14,
        "gamma": 0.96,
        "highlights": 1.2,
        "shadows": 0.9,
        "temperature": -7,
        "vibrance": 1.05
    }
}

def detect_wedding_ring_replicate(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect wedding ring using Replicate AI
    Returns: (x, y, width, height) or None if detection fails
    """
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save temporary file
        temp_path = "/tmp/temp_ring.jpg"
        pil_image.save(temp_path, quality=95)
        
        # Use object detection model
        output = replicate.run(
            "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
            input={
                "image": open(temp_path, "rb"),
                "prompt": "wedding ring, jewelry, ring",
                "threshold": 0.5
            }
        )
        
        # Parse detection results
        if output and len(output) > 0:
            # Get the first/best detection
            detection = output[0]
            x, y, w, h = detection['bbox']
            
            # Clean up
            os.remove(temp_path)
            
            return int(x), int(y), int(w), int(h)
        
        # Clean up
        os.remove(temp_path)
        
    except Exception as e:
        print(f"Replicate detection failed: {str(e)}, using fallback")
    
    return None

def detect_wedding_ring_fallback(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Fallback ring detection using OpenCV
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Threshold to find bright objects (rings are usually bright)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Morphology operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # Rings typically have aspect ratio close to 1
                if 0.5 < aspect_ratio < 2.0:
                    valid_contours.append((contour, area))
        
        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add 15% padding
            padding = int(max(w, h) * 0.15)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            return x, y, w, h
    
    # If no ring detected, return center region
    center_size = int(min(width, height) * 0.6)
    x = (width - center_size) // 2
    y = (height - center_size) // 2
    
    return x, y, center_size, center_size

def enhance_wedding_ring_advanced(image: Image.Image, metal_type: str = "white_gold") -> Image.Image:
    """
    Advanced wedding ring enhancement with detailed adjustments
    """
    params = WEDDING_RING_PARAMS.get(metal_type, WEDDING_RING_PARAMS["white_gold"])
    
    # Keep original for reference
    original = image.copy()
    
    # Convert to numpy for advanced processing
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 1. Adjust highlights and shadows
    # Create luminance channel
    luminance = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Separate highlights and shadows
    highlights_mask = np.where(luminance > 0.6, 1.0, 0.0)
    shadows_mask = np.where(luminance < 0.3, 1.0, 0.0)
    
    # Apply adjustments
    img_array = img_array * (1 + highlights_mask[:,:,np.newaxis] * (params['highlights'] - 1))
    img_array = img_array * (1 + shadows_mask[:,:,np.newaxis] * (params['shadows'] - 1))
    
    # 2. Color temperature adjustment
    temp_shift = params['temperature'] / 100.0
    img_array[:,:,0] = np.clip(img_array[:,:,0] + temp_shift, 0, 1)  # Red
    img_array[:,:,2] = np.clip(img_array[:,:,2] - temp_shift, 0, 1)  # Blue
    
    # 3. Vibrance (selective saturation)
    # Convert to HSV
    hsv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * params['vibrance']  # Adjust saturation
    img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    
    # 4. Apply gamma correction
    gamma = params['gamma']
    img_array = np.power(img_array, gamma)
    
    # 5. White overlay for metallic shine
    white_overlay = np.ones_like(img_array)
    img_array = img_array * (1 - params['white_overlay']) + white_overlay * params['white_overlay']
    
    # Convert back to PIL
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    enhanced = Image.fromarray(img_array)
    
    # 6. Apply PIL enhancements
    # Brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(params['brightness'])
    
    # Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(params['contrast'])
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(params['sharpness'])
    
    # Saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(params['saturation'])
    
    # 7. Detail enhancement
    # Unsharp mask for fine details
    gaussian = enhanced.filter(ImageFilter.GaussianBlur(radius=2))
    enhanced = Image.blend(gaussian, enhanced, 1.5)
    
    # 8. Final sharpening
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    
    return enhanced

def crop_and_upscale(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     target_size: Tuple[int, int] = (1000, 1300)) -> np.ndarray:
    """
    Crop to ring area and upscale to target size
    """
    x, y, w, h = bbox
    
    # Calculate aspect ratio for target size
    target_aspect = target_size[0] / target_size[1]  # 1000/1300 = 0.77
    current_aspect = w / h
    
    # Adjust crop to match target aspect ratio
    if current_aspect > target_aspect:
        # Too wide, increase height
        new_h = int(w / target_aspect)
        y_offset = (new_h - h) // 2
        y = max(0, y - y_offset)
        h = new_h
    else:
        # Too tall, increase width
        new_w = int(h * target_aspect)
        x_offset = (new_w - w) // 2
        x = max(0, x - x_offset)
        w = new_w
    
    # Ensure within image bounds
    height, width = image.shape[:2]
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    w = min(w, width - x)
    h = min(h, height - y)
    
    # Crop
    cropped = image[y:y+h, x:x+w]
    
    # Convert to PIL for high-quality resize
    pil_cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    # Upscale using LANCZOS
    upscaled = pil_cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    # Additional sharpening after upscale
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)

def process_wedding_ring_v107(image_base64: str, metal_type: str = "white_gold") -> Dict:
    """
    Main processing function v107 - Clean architecture
    """
    try:
        # Decode base64
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Add padding if needed
        padding = 4 - len(image_base64) % 4
        if padding != 4:
            image_base64 += '=' * padding
        
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_bytes))
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        print(f"Processing v107: metal={metal_type}, original_size={img_array.shape}")
        
        # Step 1: Detect wedding ring
        bbox = detect_wedding_ring_replicate(img_array)
        
        if bbox is None:
            print("Replicate detection failed, using fallback")
            bbox = detect_wedding_ring_fallback(img_array)
        
        print(f"Ring detected at: {bbox}")
        
        # Step 2: Crop and upscale to 1000x1300
        cropped_upscaled = crop_and_upscale(img_array, bbox, (1000, 1300))
        
        # Step 3: Convert to PIL and apply enhancement
        pil_image = Image.fromarray(cv2.cvtColor(cropped_upscaled, cv2.COLOR_BGR2RGB))
        enhanced = enhance_wedding_ring_advanced(pil_image, metal_type)
        
        # Step 4: Create thumbnail (800x800)
        thumbnail = enhanced.resize((800, 800), Image.Resampling.LANCZOS)
        thumbnail = thumbnail.filter(ImageFilter.SHARPEN)
        
        # Convert to base64
        # Main image (1000x1300)
        main_buffer = BytesIO()
        enhanced.save(main_buffer, format='PNG', quality=95, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail (800x800)
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Log sizes for debugging
        print(f"Main image size: {len(main_base64)} bytes (base64)")
        print(f"Thumbnail size: {len(thumb_base64)} bytes (base64)")
        
        # CRITICAL: Return with nested output structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "metal_type": metal_type,
                "processing_version": "v107_clean_architecture",
                "dimensions": {
                    "main": "1000x1300",
                    "thumbnail": "800x800"
                },
                "ring_detection": {
                    "method": "replicate" if bbox else "fallback",
                    "bbox": bbox
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v107: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v107_clean_architecture"
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        input_data = event.get("input", {})
        
        # Test mode check
        if input_data.get("test") == True:
            return {
                "status": "test_success",
                "message": "Wedding Ring Processor v107 - Clean Architecture",
                "version": "v107_clean_architecture",
                "features": [
                    "Direct wedding ring detection using Replicate AI",
                    "No black border detection needed",
                    "Fixed 1000x1300 main output",
                    "800x800 thumbnail output",
                    "Advanced metal-specific enhancement",
                    "Simplified and optimized workflow"
                ]
            }
        
        # Get image
        image_base64 = input_data.get("image") or input_data.get("image_base64")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error"
                }
            }
        
        # Get metal type
        metal_type = input_data.get("metal_type", "white_gold")
        if metal_type not in WEDDING_RING_PARAMS:
            metal_type = "white_gold"
        
        # Process image
        return process_wedding_ring_v107(image_base64, metal_type)
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
