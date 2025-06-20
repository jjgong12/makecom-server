import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
    Find wedding ring in image using edge detection and contour analysis
    """
    # Convert to grayscale
    gray = img.convert('L')
    gray_np = np.array(gray)
    
    # Apply edge detection
    edges = np.zeros_like(gray_np)
    for i in range(1, gray_np.shape[0]-1):
        for j in range(1, gray_np.shape[1]-1):
            gx = gray_np[i+1,j] - gray_np[i-1,j]
            gy = gray_np[i,j+1] - gray_np[i,j-1]
            edges[i,j] = min(255, abs(gx) + abs(gy))
    
    # Find connected components
    height, width = gray_np.shape
    center_x, center_y = width // 2, height // 2
    
    # Default to center area if no clear ring found
    default_size = min(width, height) // 3
    bbox = (center_x - default_size//2, center_y - default_size//2, default_size, default_size)
    
    # Look for circular patterns near center
    best_score = 0
    for y in range(height//4, 3*height//4, 10):
        for x in range(width//4, 3*width//4, 10):
            # Check if this could be ring center
            score = 0
            for r in range(20, min(width, height)//4, 5):
                edge_sum = 0
                count = 0
                for angle in range(0, 360, 10):
                    px = int(x + r * np.cos(np.radians(angle)))
                    py = int(y + r * np.sin(np.radians(angle)))
                    if 0 <= px < width and 0 <= py < height:
                        edge_sum += edges[py, px]
                        count += 1
                if count > 0:
                    score += edge_sum / count
            
            if score > best_score:
                best_score = score
                size = min(width, height) // 2
                bbox = (max(0, x - size//2), max(0, y - size//2), size, size)
    
    print(f"Ring detected at: {bbox}")
    return bbox

def enhance_wedding_ring_v110(img: Image.Image, metal_type: str) -> Image.Image:
    """
    Enhanced wedding ring processing with stronger white treatment
    """
    # 1. Base adjustments
    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.2)
    
    # 2. Metal-specific processing
    if metal_type == "yellow_gold":
        # Warm enhancement
        r, g, b = enhanced.split()
        r = r.point(lambda i: min(255, int(i * 1.05)))
        g = g.point(lambda i: min(255, int(i * 1.03)))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "rose_gold":
        # Rose tint
        r, g, b = enhanced.split()
        r = r.point(lambda i: min(255, int(i * 1.08)))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "white_gold":
        # Cool tone
        r, g, b = enhanced.split()
        b = b.point(lambda i: min(255, int(i * 1.05)))
        enhanced = Image.merge('RGB', (r, g, b))
        
    elif metal_type == "white":
        # 무도금화이트 - 더 강한 화이트 처리
        # Stronger brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1.2)  # 추가 20% 밝기
        
        # Reduce saturation more
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(0.7)  # 채도 30% 감소
        
        # Strong white overlay
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, 0.2)  # 0.1에서 0.2로 증가
        
        # Extra sharpening for white
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=3, percent=250, threshold=2))
    
    # 3. Overall enhancement
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    # 4. Sharpening
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 5. Final brightness adjustment
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(1.05)
    
    return enhanced

def create_thumbnail_v110(img: Image.Image, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = (1000, 1300)) -> Image.Image:
    """
    Create perfect thumbnail with exact 1000x1300 size
    """
    x, y, w, h = bbox
    img_width, img_height = img.size
    
    # Calculate expansion to fit target aspect ratio
    target_aspect = target_size[0] / target_size[1]  # 1000/1300 = 0.769
    
    # Expand bbox to match target aspect ratio
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Too wide, increase height
        new_h = int(w / target_aspect)
        expand_h = new_h - h
        y = max(0, y - expand_h // 2)
        h = new_h
    else:
        # Too tall, increase width
        new_w = int(h * target_aspect)
        expand_w = new_w - w
        x = max(0, x - expand_w // 2)
        w = new_w
    
    # Add padding for ring to be 85% of frame
    padding = int(max(w, h) * 0.15)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_width - x, w + 2 * padding)
    h = min(img_height - y, h + 2 * padding)
    
    # Ensure we don't exceed image boundaries
    if x + w > img_width:
        w = img_width - x
    if y + h > img_height:
        h = img_height - y
    
    # Crop the image
    cropped = img.crop((x, y, x + w, y + h))
    
    # Resize to exact target size
    thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    # Apply slight brightness boost to thumbnail
    enhancer = ImageEnhance.Brightness(thumbnail)
    thumbnail = enhancer.enhance(1.05)
    
    return thumbnail

def handler(event):
    """
    RunPod handler function for wedding ring processing v110
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
        
        print(f"Processing v110: size={img.size}")
        
        # Step 1: Detect wedding ring
        bbox = find_wedding_ring(img)
        
        # Step 2: Auto detect metal type if needed
        if metal_type == "auto":
            detected_metal = detect_metal_type_from_ring(img, bbox)
            print(f"Auto-detected metal type: {detected_metal}")
        else:
            detected_metal = metal_type
        
        # Step 3: Enhance the image
        enhanced = enhance_wedding_ring_v110(img, detected_metal)
        
        # Step 4: Create thumbnail (1000x1300)
        thumbnail = create_thumbnail_v110(enhanced, bbox)
        
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
                    "version": "v110"
                }
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v110"
            }
        }

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
