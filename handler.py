import runpod
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from io import BytesIO
import base64
import traceback
import os
import requests

# Replicate API token from environment
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

def print_debug(msg):
    """Debug helper"""
    print(f"[DEBUG v115] {msg}")

def safe_base64_decode(base64_string):
    """
    Safely decode base64 with multiple strategies
    """
    if not base64_string:
        raise ValueError("Empty base64 string")
    
    # Remove data URL prefix if present
    if base64_string.startswith('data:'):
        base64_string = base64_string.split(',')[1]
    
    # Try standard decode
    try:
        return base64.b64decode(base64_string)
    except:
        pass
    
    # Try with padding
    try:
        padding = 4 - len(base64_string) % 4
        if padding != 4:
            base64_string += '=' * padding
        return base64.b64decode(base64_string)
    except:
        pass
    
    # Try URL-safe decode
    try:
        return base64.urlsafe_b64decode(base64_string)
    except:
        pass
    
    # Last resort - clean and retry
    try:
        cleaned = base64_string.replace('-', '+').replace('_', '/')
        padding = 4 - len(cleaned) % 4
        if padding != 4:
            cleaned += '=' * padding
        return base64.b64decode(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to decode base64: {str(e)}")

def detect_black_masking(img):
    """
    Detect black masking areas in image
    """
    gray = np.array(img.convert('L'))
    h, w = gray.shape
    
    # Check edges for black borders
    edge_width = 100
    borders = {
        'top': 0,
        'bottom': h,
        'left': 0,
        'right': w,
        'has_masking': False
    }
    
    # Scan from edges
    scan_depth = int(min(h, w) * 0.4)
    threshold = 40
    
    # Top border
    for y in range(scan_depth):
        row = gray[y, :]
        if np.mean(row) > threshold:
            borders['top'] = y
            break
    
    # Bottom border
    for y in range(scan_depth):
        row = gray[h-1-y, :]
        if np.mean(row) > threshold:
            borders['bottom'] = h - y
            break
    
    # Left border
    for x in range(scan_depth):
        col = gray[:, x]
        if np.mean(col) > threshold:
            borders['left'] = x
            break
    
    # Right border
    for x in range(scan_depth):
        col = gray[:, w-1-x]
        if np.mean(col) > threshold:
            borders['right'] = w - x
            break
    
    # Check if significant masking detected
    max_border = max(borders['top'], h - borders['bottom'], 
                     borders['left'], w - borders['right'])
    borders['has_masking'] = max_border > 10
    borders['max_border'] = max_border
    
    return borders

def apply_replicate_inpainting(img, borders):
    """
    Apply Replicate AI inpainting for black masking removal
    """
    if not REPLICATE_API_TOKEN:
        print_debug("No Replicate API token, using simple inpainting")
        return apply_simple_inpainting(img, borders)
    
    try:
        import replicate
        
        # Create mask for inpainting
        h, w = img.size[1], img.size[0]
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw white areas where black masking is
        if borders['top'] > 0:
            draw.rectangle([(0, 0), (w, borders['top'])], fill=255)
        if borders['bottom'] < h:
            draw.rectangle([(0, borders['bottom']), (w, h)], fill=255)
        if borders['left'] > 0:
            draw.rectangle([(0, 0), (borders['left'], h)], fill=255)
        if borders['right'] < w:
            draw.rectangle([(borders['right'], 0), (w, h)], fill=255)
        
        # Convert to base64
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Select model based on border size
        if borders['max_border'] < 50:
            # Small borders - use faster model
            print_debug(f"Using Ideogram v2-turbo for {borders['max_border']}px borders")
            output = replicate.run(
                "ideogram-ai/ideogram-v2-turbo",
                input={
                    "prompt": "clean white seamless background for product photography, soft gradient",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "mode": "inpaint",
                    "num_inference_steps": 15,
                    "guidance_scale": 7.0
                }
            )
        else:
            # Large borders - use high quality model
            print_debug(f"Using FLUX Fill for {borders['max_border']}px borders")
            output = replicate.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": "professional product photography background, clean white studio background, seamless gradient",
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "num_inference_steps": 25,
                    "guidance_scale": 30.0
                }
            )
        
        # Get result
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = output
        
        # Download result
        response = requests.get(result_url)
        result = Image.open(BytesIO(response.content))
        
        # Ensure same size
        if result.size != img.size:
            result = result.resize(img.size, Image.Resampling.LANCZOS)
        
        print_debug("Replicate inpainting successful")
        return result
        
    except Exception as e:
        print_debug(f"Replicate failed: {str(e)}, using fallback")
        return apply_simple_inpainting(img, borders)

def apply_simple_inpainting(img, borders):
    """
    Simple fallback inpainting
    """
    img_array = np.array(img)
    h, w = img.shape[:2]
    
    # Find background color from non-masked area
    x1, y1 = borders['left'], borders['top']
    x2, y2 = borders['right'], borders['bottom']
    
    if x2 > x1 + 40 and y2 > y1 + 40:
        # Sample from just inside the borders
        margin = 20
        bg_region = img_array[y1+margin:y1+margin+50, x1+margin:x2-margin]
        if bg_region.size > 0:
            bg_color = np.median(bg_region.reshape(-1, 3), axis=0).astype(np.uint8)
        else:
            bg_color = np.array([245, 242, 238], dtype=np.uint8)
    else:
        bg_color = np.array([245, 242, 238], dtype=np.uint8)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    if borders['top'] > 0:
        mask[:borders['top'], :] = 255
    if borders['bottom'] < h:
        mask[borders['bottom']:, :] = 255
    if borders['left'] > 0:
        mask[:, :borders['left']] = 255
    if borders['right'] < w:
        mask[:, borders['right']:] = 255
    
    # Apply inpainting
    result = img_array.copy()
    result[mask > 0] = bg_color
    
    # Smooth edges
    result_pil = Image.fromarray(result)
    mask_pil = Image.fromarray(mask)
    mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Blend
    return Image.composite(result_pil, img, ImageOps.invert(mask_blurred))

def detect_ring_metal_type_conservative(img):
    """
    Conservative metal type detection - very careful with yellow gold
    """
    # Center crop for better detection
    w, h = img.size
    crop_size = min(w, h) // 2
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    center = img.crop((left, top, left + crop_size, top + crop_size))
    
    # Convert to numpy for analysis
    pixels = np.array(center)
    
    # Calculate average color
    avg_color = pixels.mean(axis=(0, 1))
    r, g, b = avg_color
    
    # Calculate color properties
    brightness = (r + g + b) / 3
    saturation = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1)
    
    # Color differences
    rg_diff = abs(r - g)
    gb_diff = abs(g - b)
    rb_diff = abs(r - b)
    
    print_debug(f"Color analysis: R={r:.1f}, G={g:.1f}, B={b:.1f}, Brightness={brightness:.1f}, Sat={saturation:.3f}")
    print_debug(f"Diffs: R-G={rg_diff:.1f}, G-B={gb_diff:.1f}, R-B={rb_diff:.1f}")
    
    # Check for white/무도금화이트 first (highest priority)
    if saturation < 0.15 and brightness > 180:
        if rg_diff < 5 and gb_diff < 5:
            return 'white_gold'
        elif r > g and g > b and rg_diff < 10:
            return 'white'  # 무도금화이트
        else:
            return 'white_gold'
    
    # Check for rose gold (second priority)
    elif r > g * 1.15 and r > b * 1.2 and rb_diff > 20:
        return 'rose_gold'
    
    # Check for white gold (third priority)
    elif saturation < 0.2 and brightness > 150 and gb_diff < 10:
        return 'white_gold'
    
    # Yellow gold only if very clear warm tone
    elif r > g and g > b and rg_diff > 8 and gb_diff > 8 and brightness < 180:
        yellow_ratio = (r + g) / (2 * b + 1)
        if yellow_ratio > 1.3:
            return 'yellow_gold'
        else:
            return 'white'  # Default to 무도금화이트
    
    # Default to 무도금화이트
    else:
        return 'white'

def find_wedding_ring_area(img):
    """
    Find wedding ring location in image
    """
    gray = np.array(img.convert('L'))
    h, w = gray.shape
    
    # Use variance to find ring area
    step = min(w, h) // 8
    best_var = 0
    best_region = None
    
    for y in range(step, h - step * 2, step//2):
        for x in range(step, w - step * 2, step//2):
            region = gray[y:y+step, x:x+step]
            if region.size > 0:
                var = np.var(region)
                if var > best_var:
                    best_var = var
                    best_region = (x, y, x + step, y + step)
    
    if best_region:
        # Expand region
        x1, y1, x2, y2 = best_region
        padding = step // 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return (x1, y1, x2, y2)
    
    # Fallback to center
    size = min(w, h) * 2 // 3
    x = (w - size) // 2
    y = (h - size) // 2
    return (x, y, x + size, y + size)

def enhance_wedding_ring_v115(img, metal_type='auto', strength=0.7):
    """
    Enhanced wedding ring processing with detail preservation
    """
    if metal_type == 'auto':
        metal_type = detect_ring_metal_type_conservative(img)
    
    print_debug(f"Processing with metal type: {metal_type}")
    
    # Find ring area
    ring_bbox = find_wedding_ring_area(img)
    
    # Base enhancements
    enhanced = img.copy()
    
    # Apply different enhancement to ring area
    if ring_bbox:
        x1, y1, x2, y2 = ring_bbox
        
        # Extract ring area
        ring_area = img.crop((x1, y1, x2, y2))
        
        # Strong enhancement for ring area
        ring_area = ImageEnhance.Sharpness(ring_area).enhance(1.5 * strength)
        ring_area = ImageEnhance.Contrast(ring_area).enhance(1.3 * strength)
        ring_area = ring_area.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
        
        # Metal-specific adjustments
        if metal_type == 'white':  # 무도금화이트
            ring_area = ImageEnhance.Brightness(ring_area).enhance(1.25 * strength)
            ring_area = ImageEnhance.Color(ring_area).enhance(0.85)
            r, g, b = ring_area.split()
            b = ImageEnhance.Brightness(b).enhance(1.03)
            ring_area = Image.merge('RGB', (r, g, b))
        
        elif metal_type == 'rose_gold':
            ring_area = ImageEnhance.Brightness(ring_area).enhance(1.15 * strength)
            ring_area = ImageEnhance.Color(ring_area).enhance(1.2 * strength)
            r, g, b = ring_area.split()
            r = ImageEnhance.Brightness(r).enhance(1.08)
            ring_area = Image.merge('RGB', (r, g, b))
        
        elif metal_type == 'yellow_gold':
            ring_area = ImageEnhance.Brightness(ring_area).enhance(1.2 * strength)
            ring_area = ImageEnhance.Color(ring_area).enhance(1.25 * strength)
            r, g, b = ring_area.split()
            r = ImageEnhance.Brightness(r).enhance(1.05)
            g = ImageEnhance.Brightness(g).enhance(1.03)
            ring_area = Image.merge('RGB', (r, g, b))
        
        else:  # white_gold
            ring_area = ImageEnhance.Brightness(ring_area).enhance(1.2 * strength)
            ring_area = ImageEnhance.Color(ring_area).enhance(0.9)
        
        # Paste back with soft blending
        mask = Image.new('L', (x2-x1, y2-y1), 255)
        draw = ImageDraw.Draw(mask)
        for i in range(10):
            draw.rectangle([(i, i), (mask.width-i-1, mask.height-i-1)], outline=255-i*20, width=1)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        
        enhanced.paste(ring_area, (x1, y1), mask)
    
    # Light enhancement for whole image
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
    
    # Final white overlay for shine
    if strength > 0.5 and metal_type in ['white', 'white_gold']:
        white_layer = Image.new('RGB', enhanced.size, (255, 255, 255))
        enhanced = Image.blend(enhanced, white_layer, 0.08 * strength)
    
    return enhanced, metal_type

def create_clean_background(img, target_color=(245, 242, 238)):
    """
    Create clean background while preserving ring
    """
    # Find ring area
    ring_bbox = find_wedding_ring_area(img)
    if not ring_bbox:
        return img
    
    x1, y1, x2, y2 = ring_bbox
    
    # Create gradient background
    w, h = img.size
    background = Image.new('RGB', (w, h), target_color)
    
    # Extract ring with soft mask
    ring_area = img.crop((x1, y1, x2, y2))
    
    # Create mask based on contrast
    gray = np.array(ring_area.convert('L'))
    threshold = np.percentile(gray, 70)
    mask = Image.fromarray((gray < threshold).astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Composite
    background.paste(ring_area, (x1, y1), mask)
    
    return background

def create_thumbnail(img, target_size=(1000, 1300)):
    """
    Create perfect thumbnail with ring centered
    """
    # Find ring
    bbox = find_wedding_ring_area(img)
    if bbox:
        x1, y1, x2, y2 = bbox
    else:
        # Center crop
        w, h = img.size
        size = min(w, h) * 3 // 4
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size
    
    # Add small padding
    padding = int((x2 - x1) * 0.1)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)
    
    # Crop
    cropped = img.crop((x1, y1, x2, y2))
    
    # Create clean background
    background = Image.new('RGB', target_size, (248, 248, 248))
    
    # Scale to fit
    scale = min(target_size[0] / cropped.width * 0.9,
                target_size[1] / cropped.height * 0.9)
    
    new_size = (int(cropped.width * scale), int(cropped.height * scale))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center paste
    x = (target_size[0] - new_size[0]) // 2
    y = (target_size[1] - new_size[1]) // 2
    
    background.paste(resized, (x, y))
    
    return background

def handler(event):
    """
    RunPod handler function for wedding ring processing v115
    """
    try:
        print_debug("Handler started")
        
        # Get input data
        input_data = event.get('input', {})
        
        # Try multiple keys for image
        image_base64 = None
        for key in ['image', 'image_base64', 'base64', 'img']:
            if key in input_data:
                image_base64 = input_data[key]
                print_debug(f"Found image in key: {key}")
                break
        
        if not image_base64 and isinstance(input_data, str):
            image_base64 = input_data
            print_debug("Input data is base64 string")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Get metal type
        metal_type = input_data.get('metal_type', 'auto') if isinstance(input_data, dict) else 'auto'
        
        # Decode base64
        try:
            img_bytes = safe_base64_decode(image_base64)
            print_debug(f"Decoded {len(img_bytes)} bytes")
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to decode base64: {str(e)}",
                    "status": "error"
                }
            }
        
        # Open image
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print_debug(f"Image size: {img.size}")
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to open image: {str(e)}",
                    "status": "error"
                }
            }
        
        # Step 1: Detect black masking
        borders = detect_black_masking(img)
        print_debug(f"Black masking detection: {borders}")
        
        # Step 2: Remove black masking with Replicate if needed
        if borders['has_masking']:
            print_debug("Applying Replicate AI inpainting")
            img_cleaned = apply_replicate_inpainting(img, borders)
            
            # Crop to content area
            if borders['top'] > 0 or borders['bottom'] < img.height or \
               borders['left'] > 0 or borders['right'] < img.width:
                img_cleaned = img_cleaned.crop((
                    borders['left'], borders['top'],
                    borders['right'], borders['bottom']
                ))
                print_debug(f"Cropped to: {img_cleaned.size}")
        else:
            img_cleaned = img
            print_debug("No masking detected, proceeding with original")
        
        # Step 3: Enhance image (light enhancement)
        enhanced, detected_metal = enhance_wedding_ring_v115(img_cleaned, metal_type, strength=0.4)
        print_debug(f"Enhanced with metal type: {detected_metal}")
        
        # Step 4: Create clean background
        enhanced_clean = create_clean_background(enhanced)
        
        # Step 5: Create thumbnail (stronger enhancement)
        img_for_thumb, _ = enhance_wedding_ring_v115(img_cleaned, detected_metal, strength=1.0)
        thumbnail = create_thumbnail(img_for_thumb)
        
        # Convert to base64
        # Enhanced image
        enhanced_buffer = BytesIO()
        enhanced_clean.save(enhanced_buffer, format='PNG', quality=95, optimize=True)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        print_debug("Processing complete")
        
        # Return with correct structure
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "metal_type": detected_metal,
                "masking_removed": borders['has_masking'],
                "inpainting_method": "replicate" if borders['has_masking'] and REPLICATE_API_TOKEN else "simple",
                "original_size": f"{img.width}x{img.height}",
                "processing_version": "v115_replicate_inpainting",
                "status": "success"
            }
        }
        
    except Exception as e:
        print_debug(f"Error: {str(e)}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
