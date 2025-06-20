import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os
import requests

def detect_metal_type_from_ring(img: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
    """
    Detect metal type from wedding ring area with improved accuracy
    """
    x, y, w, h = bbox
    
    # Crop ring area for analysis
    ring_area = img.crop((x, y, x+w, y+h))
    
    # Convert to numpy for analysis
    ring_np = np.array(ring_area)
    
    # Calculate average color in RGB
    avg_r = np.mean(ring_np[:,:,0])
    avg_g = np.mean(ring_np[:,:,1])
    avg_b = np.mean(ring_np[:,:,2])
    
    # Calculate brightness and saturation
    brightness = (avg_r + avg_g + avg_b) / 3
    max_val = max(avg_r, avg_g, avg_b)
    min_val = min(avg_r, avg_g, avg_b)
    saturation = (max_val - min_val) / max_val if max_val > 0 else 0
    
    print(f"Metal detection - RGB:({avg_r:.1f},{avg_g:.1f},{avg_b:.1f}), Brightness:{brightness:.1f}, Saturation:{saturation:.3f}")
    
    # Improved detection logic
    # 무도금화이트 감지 (매우 낮은 채도, 높은 밝기)
    if saturation < 0.15 and brightness > 180:
        return "white"
    
    # 로즈골드 감지 (붉은색 우세)
    if avg_r > avg_g + 15 and avg_r > avg_b + 20:
        return "rose_gold"
    
    # 화이트골드 감지 (차가운 톤, 중간 채도)
    if avg_b > avg_r and saturation < 0.25:
        return "white_gold"
    
    # 옐로우골드 감지 (따뜻한 황금색)
    if avg_g > avg_b and avg_r > avg_b and saturation > 0.2:
        return "yellow_gold"
    
    # 기본값: 화이트골드
    return "white_gold"

def find_wedding_ring(img: Image.Image) -> Tuple[int, int, int, int]:
    """
    Enhanced wedding ring detection for various backgrounds
    """
    width, height = img.size
    img_np = np.array(img)
    
    # Convert to grayscale
    gray = img.convert('L')
    gray_np = np.array(gray)
    
    # Apply edge detection with Sobel filter
    edges = np.zeros_like(gray_np, dtype=float)
    for i in range(1, gray_np.shape[0]-1):
        for j in range(1, gray_np.shape[1]-1):
            # Sobel X
            gx = (gray_np[i-1,j+1] + 2*gray_np[i,j+1] + gray_np[i+1,j+1] -
                  gray_np[i-1,j-1] - 2*gray_np[i,j-1] - gray_np[i+1,j-1])
            # Sobel Y
            gy = (gray_np[i+1,j-1] + 2*gray_np[i+1,j] + gray_np[i+1,j+1] -
                  gray_np[i-1,j-1] - 2*gray_np[i-1,j] - gray_np[i-1,j+1])
            edges[i,j] = np.sqrt(gx**2 + gy**2)
    
    # Normalize edges
    edges = (edges / edges.max() * 255).astype(np.uint8)
    
    # Find regions with high edge density (likely rings)
    center_x, center_y = width // 2, height // 2
    best_bbox = None
    best_score = 0
    
    # Search in multiple scales
    for scale in [0.2, 0.3, 0.4, 0.5]:
        search_size = int(min(width, height) * scale)
        
        # Scan around center
        for dy in range(-height//4, height//4, 20):
            for dx in range(-width//4, width//4, 20):
                cx = center_x + dx
                cy = center_y + dy
                
                # Skip if out of bounds
                if cx - search_size//2 < 0 or cx + search_size//2 > width:
                    continue
                if cy - search_size//2 < 0 or cy + search_size//2 > height:
                    continue
                
                # Extract region
                x1 = cx - search_size//2
                y1 = cy - search_size//2
                x2 = x1 + search_size
                y2 = y1 + search_size
                
                region = edges[y1:y2, x1:x2]
                
                # Calculate edge density
                edge_density = np.sum(region > 50) / region.size
                
                # Calculate metallic score (high variance in original image)
                region_color = img_np[y1:y2, x1:x2]
                metallic_score = np.std(region_color) / 255.0
                
                # Combined score
                score = edge_density * metallic_score
                
                if score > best_score:
                    best_score = score
                    best_bbox = (x1, y1, search_size, search_size)
    
    # If no good detection, use center area
    if best_bbox is None:
        size = min(width, height) // 3
        best_bbox = (center_x - size//2, center_y - size//2, size, size)
    
    print(f"Ring detected at: {best_bbox} with score: {best_score:.3f}")
    return best_bbox

def enhance_wedding_ring_v111(img: Image.Image, metal_type: str, strength: float = 1.0) -> Image.Image:
    """
    Enhanced wedding ring processing with adjustable strength
    strength: 1.0 for full enhancement (thumbnail), 0.3 for light enhancement (main image)
    """
    # 1. Base adjustments
    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(1 + 0.15 * strength)
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1 + 0.2 * strength)
    
    # 2. Metal-specific processing
    if metal_type == "yellow_gold":
        # Warm enhancement
        r, g, b = enhanced.split()
        r = r.point(lambda i: min(255, int(i * (1 + 0.05 * strength))))
        g = g.point(lambda i: min(255, int(i * (1 + 0.03 * strength))))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "rose_gold":
        # Rose tint
        r, g, b = enhanced.split()
        r = r.point(lambda i: min(255, int(i * (1 + 0.08 * strength))))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "white_gold":
        # Cool tone
        r, g, b = enhanced.split()
        b = b.point(lambda i: min(255, int(i * (1 + 0.05 * strength))))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "white":
        # 무도금화이트 - 더 강한 화이트 처리
        # Stronger brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1 + 0.2 * strength)
        
        # Reduce saturation more
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1 - 0.3 * strength)
        
        # Strong white overlay
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, 0.2 * strength)
        
        # Extra sharpening for white
        if strength > 0.5:
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=3, percent=int(250 * strength), threshold=2))
    
    # 3. Overall enhancement
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1 + 0.1 * strength)
    
    # 4. Sharpening
    if strength > 0.5:
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150 * strength), threshold=3))
    
    # 5. Final brightness adjustment
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(1 + 0.05 * strength)
    
    return enhanced

def create_thumbnail_v111(img: Image.Image, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = (1000, 1300)) -> Image.Image:
    """
    Create perfect thumbnail with clean background and proper ring centering
    """
    x, y, w, h = bbox
    img_width, img_height = img.size
    
    # Calculate center of the ring
    ring_center_x = x + w // 2
    ring_center_y = y + h // 2
    
    # Determine crop size based on target aspect ratio
    target_aspect = target_size[0] / target_size[1]  # 1000/1300 = 0.769
    
    # Make the crop larger to ensure ring fits well
    crop_size = int(max(w, h) * 1.4)  # 40% padding around the ring
    
    # Adjust crop size to match aspect ratio
    if target_aspect < 1:  # Portrait
        crop_height = crop_size
        crop_width = int(crop_height * target_aspect)
    else:  # Landscape
        crop_width = crop_size
        crop_height = int(crop_width / target_aspect)
    
    # Calculate crop coordinates centered on ring
    crop_x1 = max(0, ring_center_x - crop_width // 2)
    crop_y1 = max(0, ring_center_y - crop_height // 2)
    crop_x2 = min(img_width, crop_x1 + crop_width)
    crop_y2 = min(img_height, crop_y1 + crop_height)
    
    # Adjust if crop exceeds image boundaries
    if crop_x2 - crop_x1 < crop_width:
        if crop_x1 == 0:
            crop_x2 = crop_width
        else:
            crop_x1 = img_width - crop_width
    
    if crop_y2 - crop_y1 < crop_height:
        if crop_y1 == 0:
            crop_y2 = crop_height
        else:
            crop_y1 = img_height - crop_height
    
    # Crop the image
    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Create clean background effect using PIL
    # Apply light background overlay to non-ring areas
    cropped_np = np.array(cropped)
    gray = np.array(cropped.convert('L'))
    
    # Find ring area (usually darker/more contrast)
    threshold = np.percentile(gray, 70)
    
    # Create soft mask
    mask = Image.fromarray(gray)
    mask = mask.point(lambda p: 255 if p < threshold else 0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Create light background
    background = Image.new('RGB', cropped.size, (248, 248, 248))
    
    # Composite ring over background
    result = Image.composite(cropped, background, mask)
    
    # Resize to exact target size with high quality
    thumbnail = result.resize(target_size, Image.Resampling.LANCZOS)
    
    return thumbnail

def handler(event):
    """
    RunPod handler function for wedding ring processing v111
    """
    try:
        # Parse input
        input_data = event.get('input', {})
        image_base64 = input_data.get('image', '')
        metal_type = input_data.get('metal_type', 'auto')
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Handle data URL format
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        try:
            img_bytes = base64.b64decode(image_base64)
        except:
            # Add padding if needed
            padding = 4 - len(image_base64) % 4
            if padding != 4:
                image_base64 += '=' * padding
            img_bytes = base64.b64decode(image_base64)
        
        # Open image
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"Processing v111: size={img.size}")
        
        # Step 1: Detect wedding ring
        bbox = find_wedding_ring(img)
        
        # Step 2: Auto detect metal type if needed
        if metal_type == "auto":
            detected_metal = detect_metal_type_from_ring(img, bbox)
            print(f"Auto-detected metal type: {detected_metal}")
        else:
            detected_metal = metal_type
        
        # Step 3: Light enhancement for main image (strength=0.3)
        enhanced = enhance_wedding_ring_v111(img, detected_metal, strength=0.3)
        
        # Step 4: Strong enhancement for thumbnail (strength=1.0)
        enhanced_for_thumb = enhance_wedding_ring_v111(img, detected_metal, strength=1.0)
        
        # Step 5: Create thumbnail (1000x1300)
        thumbnail = create_thumbnail_v111(enhanced_for_thumb, bbox)
        
        # Convert to base64
        # Enhanced full image
        buffer = BytesIO()
        enhanced.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Thumbnail
        buffer = BytesIO()
        thumbnail.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Return with correct structure
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": detected_metal,
                    "original_size": img.size,
                    "thumbnail_size": thumbnail.size,
                    "ring_bbox": bbox,
                    "status": "success",
                    "version": "v111"
                }
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v111"
            }
        }

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
