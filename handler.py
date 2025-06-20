import runpod
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from io import BytesIO
import base64
import traceback
import json

def print_debug(msg):
    """Debug helper"""
    print(f"[DEBUG v113] {msg}")

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

def detect_ring_metal_type(img):
    """
    Detect metal type from ring image
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
    
    # Detect metal type
    if saturation < 0.1 and brightness > 200:
        # Low saturation + high brightness = white/silver
        return 'white'
    elif r > g * 1.2 and r > b * 1.2:
        # Red dominant = rose gold
        return 'rose_gold'
    elif r > g and g > b and brightness < 180:
        # Warm tone = yellow gold
        return 'yellow_gold'
    else:
        # Default to white gold
        return 'white_gold'

def find_wedding_ring_multi_method(img):
    """
    Find wedding ring using multiple detection methods
    """
    gray = np.array(img.convert('L'))
    h, w = gray.shape
    
    methods = []
    
    # Method 1: Edge-based detection
    try:
        from PIL import ImageFilter
        edges = img.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges.convert('L'))
        
        # Find regions with high edge density
        regions = []
        step = min(w, h) // 10
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                region = edge_array[y:y+step, x:x+step]
                if region.size > 0:
                    density = np.mean(region)
                    regions.append((density, x, y, step, step))
        
        if regions:
            regions.sort(reverse=True)
            _, x, y, rw, rh = regions[0]
            methods.append(('edge', x, y, rw, rh))
    except:
        pass
    
    # Method 2: Variance-based detection
    try:
        step = min(w, h) // 8
        best_var = 0
        best_region = None
        
        for y in range(0, h - step, step//2):
            for x in range(0, w - step, step//2):
                region = gray[y:y+step, x:x+step]
                if region.size > 0:
                    var = np.var(region)
                    if var > best_var:
                        best_var = var
                        best_region = (x, y, step, step)
        
        if best_region:
            methods.append(('variance', *best_region))
    except:
        pass
    
    # Method 3: Center-weighted detection
    try:
        cx, cy = w // 2, h // 2
        size = min(w, h) // 2
        methods.append(('center', cx - size//2, cy - size//2, size, size))
    except:
        pass
    
    # Choose best method or fallback to center
    if methods:
        # Prefer edge detection if available
        for method in methods:
            if method[0] == 'edge':
                _, x, y, bw, bh = method
                return (x, y, x + bw, y + bh)
        # Otherwise use first available
        _, x, y, bw, bh = methods[0]
        return (x, y, x + bw, y + bh)
    
    # Final fallback: center crop
    size = min(w, h) * 3 // 4
    x = (w - size) // 2
    y = (h - size) // 2
    return (x, y, x + size, y + size)

def enhance_wedding_ring_v113(img, metal_type='auto', strength=0.7):
    """
    Enhanced wedding ring processing with metal-specific adjustments
    """
    if metal_type == 'auto':
        metal_type = detect_ring_metal_type(img)
    
    print_debug(f"Detected metal type: {metal_type}")
    
    # Base enhancements
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    
    # Metal-specific adjustments
    if metal_type == 'white':
        # White/Silver metals - increase brightness significantly
        img = ImageEnhance.Brightness(img).enhance(1.3 * strength)
        img = ImageEnhance.Color(img).enhance(0.8)
        
        # Add subtle blue tint for white metals
        if strength > 0.5:
            r, g, b = img.split()
            b = ImageEnhance.Brightness(b).enhance(1.05)
            img = Image.merge('RGB', (r, g, b))
    
    elif metal_type == 'rose_gold':
        # Rose gold - enhance reds and warmth
        img = ImageEnhance.Brightness(img).enhance(1.15 * strength)
        img = ImageEnhance.Color(img).enhance(1.2 * strength)
        
        # Enhance red channel
        if strength > 0.5:
            r, g, b = img.split()
            r = ImageEnhance.Brightness(r).enhance(1.1)
            img = Image.merge('RGB', (r, g, b))
    
    elif metal_type == 'yellow_gold':
        # Yellow gold - warm enhancement
        img = ImageEnhance.Brightness(img).enhance(1.2 * strength)
        img = ImageEnhance.Color(img).enhance(1.3 * strength)
        
        # Add warmth
        if strength > 0.5:
            r, g, b = img.split()
            r = ImageEnhance.Brightness(r).enhance(1.05)
            g = ImageEnhance.Brightness(g).enhance(1.05)
            img = Image.merge('RGB', (r, g, b))
    
    else:  # white_gold
        # White gold - balanced enhancement
        img = ImageEnhance.Brightness(img).enhance(1.2 * strength)
        img = ImageEnhance.Color(img).enhance(0.9)
    
    # Final clarity enhancement
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Add subtle white overlay for shine effect
    if strength > 0.5:
        white_layer = Image.new('RGB', img.size, (255, 255, 255))
        img = Image.blend(img, white_layer, 0.05 * strength)
    
    return img, metal_type

def create_thumbnail(img, target_size=(1000, 1300)):
    """
    Create a perfect thumbnail with wedding ring centered
    """
    # Find ring location
    bbox = find_wedding_ring_multi_method(img)
    x1, y1, x2, y2 = bbox
    
    # Add padding around detected ring
    padding = int((x2 - x1) * 0.2)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)
    
    # Crop to ring area
    cropped = img.crop((x1, y1, x2, y2))
    
    # Create clean background
    background = Image.new('RGB', target_size, (248, 248, 248))
    
    # Calculate scaling to fit
    scale = min(target_size[0] / cropped.width * 0.85,
                target_size[1] / cropped.height * 0.85)
    
    new_size = (int(cropped.width * scale), int(cropped.height * scale))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center on background
    x = (target_size[0] - new_size[0]) // 2
    y = (target_size[1] - new_size[1]) // 2
    
    # Create soft mask for blending
    mask = Image.new('L', new_size, 0)
    mask.paste(255, (0, 0, new_size[0], new_size[1]))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Composite
    background.paste(resized, (x, y), mask)
    
    return background

def handler(event):
    """
    RunPod handler function for wedding ring processing v113
    """
    try:
        print_debug("Handler started")
        print_debug(f"Event type: {type(event)}")
        print_debug(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
        
        # Get input data with extensive checking
        input_data = event.get('input', {})
        print_debug(f"Input data type: {type(input_data)}")
        print_debug(f"Input data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Try all possible keys for image
        image_base64 = None
        possible_keys = ['image', 'image_base64', 'base64', 'img', 'data', 'imageData', 'image_data']
        
        for key in possible_keys:
            if key in input_data:
                image_base64 = input_data[key]
                print_debug(f"Found image in key: {key}")
                break
        
        # Check if input_data itself is the base64 string
        if not image_base64 and isinstance(input_data, str):
            image_base64 = input_data
            print_debug("Input data itself is base64 string")
        
        # Debug what we found
        if image_base64:
            print_debug(f"Base64 length: {len(image_base64)}")
            print_debug(f"Base64 preview: {image_base64[:100]}...")
        else:
            print_debug("No image found!")
            print_debug(f"Full input_data: {json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)[:500]}")
        
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided in input",
                    "status": "error",
                    "debug_info": {
                        "event_keys": list(event.keys()) if isinstance(event, dict) else "event not dict",
                        "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else "input not dict",
                        "input_type": str(type(input_data)),
                        "checked_keys": possible_keys
                    }
                }
            }
        
        # Get metal type
        metal_type = input_data.get('metal_type', 'auto') if isinstance(input_data, dict) else 'auto'
        print_debug(f"Metal type: {metal_type}")
        
        # Safe decode base64
        try:
            img_bytes = safe_base64_decode(image_base64)
            print_debug(f"Successfully decoded {len(img_bytes)} bytes")
        except Exception as e:
            print_debug(f"Failed to decode base64: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to decode base64: {str(e)}",
                    "status": "error",
                    "base64_length": len(image_base64),
                    "base64_preview": image_base64[:100] + "..."
                }
            }
        
        # Open image
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print_debug(f"Opened image: {img.size}, mode: {img.mode}")
        except Exception as e:
            print_debug(f"Failed to open image: {str(e)}")
            return {
                "output": {
                    "error": f"Failed to open image: {str(e)}",
                    "status": "error"
                }
            }
        
        # Process enhanced image (lighter enhancement)
        enhanced, detected_metal = enhance_wedding_ring_v113(img, metal_type, strength=0.3)
        print_debug("Created enhanced image")
        
        # Create thumbnail (stronger enhancement)
        img_for_thumb, _ = enhance_wedding_ring_v113(img, detected_metal, strength=1.0)
        thumbnail = create_thumbnail(img_for_thumb)
        print_debug("Created thumbnail")
        
        # Convert to base64
        # Enhanced image
        enhanced_buffer = BytesIO()
        enhanced.save(enhanced_buffer, format='PNG', quality=95, optimize=True)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', quality=95, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        print_debug(f"Enhanced size: {len(enhanced_base64)}, Thumbnail size: {len(thumb_base64)}")
        
        # Return with correct structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumb_base64,
                "metal_type": detected_metal,
                "original_size": f"{img.width}x{img.height}",
                "processing_version": "v113_complete_fix",
                "status": "success"
            }
        }
        
    except Exception as e:
        print_debug(f"Handler error: {str(e)}")
        print_debug(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v113_complete_fix"
            }
        }

# RunPod entry point
runpod.serverless.start({"handler": handler})
