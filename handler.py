import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from typing import Dict, Tuple, Optional, List
import json
import os
import requests

def safe_base64_decode(image_base64: str) -> bytes:
    """
    Safely decode base64 with multiple fallback strategies
    """
    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[-1]
    
    # Clean base64 string
    image_base64 = image_base64.strip()
    image_base64 = image_base64.replace(' ', '+')
    image_base64 = image_base64.replace('\n', '')
    image_base64 = image_base64.replace('\r', '')
    
    # Try multiple decoding strategies
    strategies = [
        lambda s: base64.b64decode(s),
        lambda s: base64.b64decode(s + '='),
        lambda s: base64.b64decode(s + '=='),
        lambda s: base64.b64decode(s + '==='),
        lambda s: base64.b64decode(s + '===='),
    ]
    
    for strategy in strategies:
        try:
            return strategy(image_base64)
        except:
            continue
    
    # Last resort: add padding based on length
    padding = 4 - len(image_base64) % 4
    if padding != 4:
        image_base64 += '=' * padding
    
    return base64.b64decode(image_base64)

def detect_metal_type_from_ring(img: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
    """
    Detect metal type from wedding ring area with improved accuracy
    """
    x, y, w, h = bbox
    
    # Ensure bbox is within image bounds
    img_width, img_height = img.size
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
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

def find_wedding_ring_multi_method(img: Image.Image) -> Tuple[int, int, int, int]:
    """
    Multi-method wedding ring detection with fallback strategies
    """
    width, height = img.size
    
    # Method 1: Edge-based detection
    bbox1 = find_wedding_ring_by_edges(img)
    
    # Method 2: Color variance detection
    bbox2 = find_wedding_ring_by_variance(img)
    
    # Method 3: Center-weighted detection
    bbox3 = find_wedding_ring_center_weighted(img)
    
    # Combine results - choose the one closest to center with reasonable size
    candidates = [bbox1, bbox2, bbox3]
    center_x, center_y = width // 2, height // 2
    
    best_bbox = None
    best_score = float('inf')
    
    for bbox in candidates:
        if bbox is None:
            continue
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        
        # Score based on distance from center and reasonable size
        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        size_score = abs(w * h - (width * height * 0.1))  # Prefer ~10% of image area
        
        score = distance + size_score * 0.001
        
        if score < best_score:
            best_score = score
            best_bbox = bbox
    
    # If all methods fail, use safe center crop
    if best_bbox is None:
        size = min(width, height) // 3
        best_bbox = (center_x - size//2, center_y - size//2, size, size)
    
    print(f"Ring detected at: {best_bbox}")
    return best_bbox

def find_wedding_ring_by_edges(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Edge-based wedding ring detection
    """
    width, height = img.size
    img_np = np.array(img)
    
    # Convert to grayscale
    gray = img.convert('L')
    gray_np = np.array(gray)
    
    # Apply Sobel edge detection
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
    if edges.max() > 0:
        edges = (edges / edges.max() * 255).astype(np.uint8)
    else:
        return None
    
    # Find regions with high edge density
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
                edge_density = np.sum(region > 50) / region.size if region.size > 0 else 0
                
                # Calculate metallic score
                region_color = img_np[y1:y2, x1:x2]
                metallic_score = np.std(region_color) / 255.0
                
                # Combined score
                score = edge_density * metallic_score
                
                if score > best_score:
                    best_score = score
                    best_bbox = (x1, y1, search_size, search_size)
    
    return best_bbox

def find_wedding_ring_by_variance(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Variance-based detection for metallic surfaces
    """
    width, height = img.size
    img_np = np.array(img)
    
    # Calculate local variance
    center_x, center_y = width // 2, height // 2
    best_bbox = None
    best_score = 0
    
    for scale in [0.15, 0.25, 0.35]:
        size = int(min(width, height) * scale)
        
        for dy in range(-height//6, height//6, 15):
            for dx in range(-width//6, width//6, 15):
                cx = center_x + dx
                cy = center_y + dy
                
                x1 = max(0, cx - size//2)
                y1 = max(0, cy - size//2)
                x2 = min(width, x1 + size)
                y2 = min(height, y1 + size)
                
                if x2 - x1 < size * 0.8 or y2 - y1 < size * 0.8:
                    continue
                
                region = img_np[y1:y2, x1:x2]
                
                # Calculate variance in each channel
                var_r = np.var(region[:,:,0])
                var_g = np.var(region[:,:,1])
                var_b = np.var(region[:,:,2])
                
                # Metallic surfaces have high variance
                total_var = var_r + var_g + var_b
                
                if total_var > best_score:
                    best_score = total_var
                    best_bbox = (x1, y1, x2-x1, y2-y1)
    
    return best_bbox

def find_wedding_ring_center_weighted(img: Image.Image) -> Tuple[int, int, int, int]:
    """
    Simple center-weighted detection as fallback
    """
    width, height = img.size
    
    # Assume ring is near center, use different sizes
    center_x, center_y = width // 2, height // 2
    
    # Try different sizes, prefer smaller (rings are usually small in frame)
    for scale in [0.25, 0.3, 0.35, 0.4]:
        size = int(min(width, height) * scale)
        x = center_x - size // 2
        y = center_y - size // 2
        
        # Ensure within bounds
        if x >= 0 and y >= 0 and x + size <= width and y + size <= height:
            return (x, y, size, size)
    
    # Last resort
    size = min(width, height) // 3
    return (center_x - size//2, center_y - size//2, size, size)

def enhance_wedding_ring_v112(img: Image.Image, metal_type: str, strength: float = 1.0) -> Image.Image:
    """
    Enhanced wedding ring processing with adjustable strength
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
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1 + 0.2 * strength)
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1 - 0.3 * strength)
        
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, 0.2 * strength)
        
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

def create_thumbnail_v112(img: Image.Image, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = (1000, 1300)) -> Image.Image:
    """
    Create perfect thumbnail with clean background and proper ring centering
    """
    x, y, w, h = bbox
    img_width, img_height = img.size
    
    # Ensure bbox is within bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    # Calculate center of the ring
    ring_center_x = x + w // 2
    ring_center_y = y + h // 2
    
    # Determine crop size based on target aspect ratio
    target_aspect = target_size[0] / target_size[1]  # 0.769
    
    # Make the crop larger to ensure ring fits well
    crop_size = int(max(w, h) * 1.4)
    
    # Adjust crop size to match aspect ratio
    if target_aspect < 1:
        crop_height = crop_size
        crop_width = int(crop_height * target_aspect)
    else:
        crop_width = crop_size
        crop_height = int(crop_width / target_aspect)
    
    # Calculate crop coordinates centered on ring
    crop_x1 = max(0, ring_center_x - crop_width // 2)
    crop_y1 = max(0, ring_center_y - crop_height // 2)
    crop_x2 = min(img_width, crop_x1 + crop_width)
    crop_y2 = min(img_height, crop_y1 + crop_height)
    
    # Adjust if crop exceeds image boundaries
    actual_width = crop_x2 - crop_x1
    actual_height = crop_y2 - crop_y1
    
    if actual_width < crop_width or actual_height < crop_height:
        # Create a white canvas and paste the available crop
        canvas = Image.new('RGB', (crop_width, crop_height), (248, 248, 248))
        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        paste_x = (crop_width - actual_width) // 2
        paste_y = (crop_height - actual_height) // 2
        canvas.paste(cropped, (paste_x, paste_y))
        cropped = canvas
    else:
        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Create clean background effect
    cropped_np = np.array(cropped)
    gray = np.array(cropped.convert('L'))
    
    # Find ring area
    threshold = np.percentile(gray, 70)
    
    # Create soft mask
    mask = Image.fromarray(gray)
    mask = mask.point(lambda p: 255 if p < threshold else 0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Create light background
    background = Image.new('RGB', cropped.size, (248, 248, 248))
    
    # Composite ring over background
    result = Image.composite(cropped, background, mask)
    
    # Resize to exact target size
    thumbnail = result.resize(target_size, Image.Resampling.LANCZOS)
    
    return thumbnail

def handler(event):
    """
    RunPod handler function for wedding ring processing v112
    """
    try:
        # Parse input with multiple fallbacks
        input_data = event.get('input', {})
        
        # Try different keys for image
        image_base64 = input_data.get('image', '')
        if not image_base64:
            image_base64 = input_data.get('image_base64', '')
        if not image_base64:
            image_base64 = input_data.get('base64', '')
        if not image_base64:
            # Check if input_data itself is the base64 string
            if isinstance(input_data, str):
                image_base64 = input_data
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input. Expected keys: 'image', 'image_base64', or 'base64'",
                    "status": "error",
                    "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "input is not dict"
                }
            }
        
        metal_type = input_data.get('metal_type', 'auto') if isinstance(input_data, dict) else 'auto'
        
        # Safe decode base64
        try:
            img_bytes = safe_base64_decode(image_base64)
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to decode base64: {str(e)}",
                    "status": "error",
                    "base64_length": len(image_base64),
                    "base64_preview": image_base64[:50] + "..."
                }
            }
        
        # Open image
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to open image: {str(e)}",
                    "status": "error"
                }
            }
        
        print(f"Processing v112: size={img.size}")
        
        # Step 1: Detect wedding ring with multi-method approach
        bbox = find_wedding_ring_multi_method(img)
        
        # Step 2: Auto detect metal type if needed
        if metal_type == "auto":
            detected_metal = detect_metal_type_from_ring(img, bbox)
            print(f"Auto-detected metal type: {detected_metal}")
        else:
            detected_metal = metal_type
        
        # Step 3: Light enhancement for main image (strength=0.3)
        enhanced = enhance_wedding_ring_v112(img, detected_metal, strength=0.3)
        
        # Step 4: Strong enhancement for thumbnail (strength=1.0)
        enhanced_for_thumb = enhance_wedding_ring_v112(img, detected_metal, strength=1.0)
        
        # Step 5: Create thumbnail (1000x1300)
        thumbnail = create_thumbnail_v112(enhanced_for_thumb, bbox)
        
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
                    "version": "v112"
                }
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "v112",
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
