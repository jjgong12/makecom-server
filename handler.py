import runpod
import base64
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from io import BytesIO
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import traceback

def detect_black_border_ultimate(image_array: np.ndarray) -> Tuple[int, int, int, int]:
    """Ultimate precision black border detection with multi-method approach"""
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Multi-color space detection
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
    
    # Initialize crops
    top_crop = 0
    bottom_crop = h
    left_crop = 0
    right_crop = w
    
    # Method 1: Connected Component Analysis
    # Find large black regions
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to connect nearby black pixels
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = 255 - binary  # Invert for connected components
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Find border-touching components
    for i in range(1, num_labels):
        x, y, w_comp, h_comp, area = stats[i]
        
        # Check if component touches image border
        if area > 100:  # Minimum area threshold
            if x == 0:  # Touches left border
                left_crop = max(left_crop, x + w_comp)
            if x + w_comp >= w:  # Touches right border
                right_crop = min(right_crop, x)
            if y == 0:  # Touches top border
                top_crop = max(top_crop, y + h_comp)
            if y + h_comp >= h:  # Touches bottom border
                bottom_crop = min(bottom_crop, y)
    
    # Method 2: Line continuity check
    # Check for continuous black lines
    def check_line_continuity(line, threshold=60):
        """Check if a line has continuous black pixels"""
        if len(line.shape) == 1:  # Grayscale
            black_pixels = line < threshold
        else:  # Color
            black_pixels = np.mean(line, axis=-1) < threshold
        
        # Find longest continuous black segment
        max_continuous = 0
        current_continuous = 0
        
        for is_black in black_pixels:
            if is_black:
                current_continuous += 1
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0
        
        # If more than 80% of line is continuous black
        return max_continuous > len(line) * 0.8
    
    # Scan from all directions with continuity check
    # Top scan
    for i in range(min(h//2, 300)):
        if check_line_continuity(gray[i, :]):
            top_crop = i + 1
        else:
            break
    
    # Bottom scan
    for i in range(min(h//2, 300)):
        if check_line_continuity(gray[h-1-i, :]):
            bottom_crop = h - 1 - i
        else:
            break
    
    # Left scan
    for i in range(min(w//2, 300)):
        if check_line_continuity(gray[:, i]):
            left_crop = i + 1
        else:
            break
    
    # Right scan
    for i in range(min(w//2, 300)):
        if check_line_continuity(gray[:, w-1-i]):
            right_crop = w - 1 - i
        else:
            break
    
    # Method 3: Diagonal scan for corner detection
    # Check diagonals to detect rounded corners
    corner_size = 100
    
    # Top-left corner
    for i in range(min(corner_size, h//4, w//4)):
        diagonal = gray[i, i]
        if diagonal < 80:
            top_crop = max(top_crop, i + 1)
            left_crop = max(left_crop, i + 1)
    
    # Top-right corner
    for i in range(min(corner_size, h//4, w//4)):
        diagonal = gray[i, w-1-i]
        if diagonal < 80:
            top_crop = max(top_crop, i + 1)
            right_crop = min(right_crop, w - 1 - i)
    
    # Bottom-left corner
    for i in range(min(corner_size, h//4, w//4)):
        diagonal = gray[h-1-i, i]
        if diagonal < 80:
            bottom_crop = min(bottom_crop, h - 1 - i)
            left_crop = max(left_crop, i + 1)
    
    # Bottom-right corner
    for i in range(min(corner_size, h//4, w//4)):
        diagonal = gray[h-1-i, w-1-i]
        if diagonal < 80:
            bottom_crop = min(bottom_crop, h - 1 - i)
            right_crop = min(right_crop, w - 1 - i)
    
    # Method 4: Multi-threshold progressive scan
    thresholds = [20, 40, 60, 80, 100, 120]
    
    for threshold in thresholds:
        # Create mask for current threshold
        mask = gray < threshold
        
        # Check edges
        edge_thickness = 5
        
        # Check if edges are mostly black
        if np.mean(mask[:edge_thickness, :]) > 0.8:
            for i in range(edge_thickness, min(h//3, 200)):
                if np.mean(mask[i, :]) > 0.7:
                    top_crop = max(top_crop, i + 1)
                else:
                    break
        
        if np.mean(mask[-edge_thickness:, :]) > 0.8:
            for i in range(edge_thickness, min(h//3, 200)):
                if np.mean(mask[h-1-i, :]) > 0.7:
                    bottom_crop = min(bottom_crop, h - 1 - i)
                else:
                    break
        
        if np.mean(mask[:, :edge_thickness]) > 0.8:
            for i in range(edge_thickness, min(w//3, 200)):
                if np.mean(mask[:, i]) > 0.7:
                    left_crop = max(left_crop, i + 1)
                else:
                    break
        
        if np.mean(mask[:, -edge_thickness:]) > 0.8:
            for i in range(edge_thickness, min(w//3, 200)):
                if np.mean(mask[:, w-1-i]) > 0.7:
                    right_crop = min(right_crop, w - 1 - i)
                else:
                    break
    
    # Method 5: Color space detection (HSV and LAB)
    # Detect very dark regions in multiple color spaces
    
    # HSV detection - low value indicates darkness
    v_channel = hsv[:, :, 2]
    
    # LAB detection - low L indicates darkness
    l_channel = lab[:, :, 0]
    
    # Combined darkness mask
    dark_mask = (v_channel < 80) & (l_channel < 80)
    
    # Apply same edge detection logic with combined mask
    for i in range(min(h//3, 200)):
        if np.mean(dark_mask[i, :]) > 0.8:
            top_crop = max(top_crop, i + 1)
        else:
            break
    
    # Minimum crop guarantee (at least 30 pixels from each edge)
    min_crop = 30
    top_crop = max(top_crop, min_crop)
    left_crop = max(left_crop, min_crop)
    bottom_crop = min(bottom_crop, h - min_crop)
    right_crop = min(right_crop, w - min_crop)
    
    # Ensure valid crops
    if top_crop >= bottom_crop - 100 or left_crop >= right_crop - 100:
        # Fallback to aggressive but safe crop
        return min_crop, min_crop, h - min_crop, w - min_crop
    
    return top_crop, left_crop, bottom_crop, right_crop

def detect_ring_mask(image_array: np.ndarray) -> np.ndarray:
    """Create mask for wedding ring with precision detection"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Use multiple detection methods
    # 1. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 2. Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Circle detection for rings
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=30, maxRadius=300)
    
    # Create combined mask
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add adaptive threshold regions
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < h * w * 0.5:  # Ring-sized objects
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Add edge-based regions
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.bitwise_or(mask, edge_dilated)
    
    # Add circle regions if found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(mask, (circle[0], circle[1]), circle[2] + 20, 255, -1)
    
    # Focus on center 70% area where ring typically is
    center_mask = np.zeros_like(mask)
    cv2.rectangle(center_mask, 
                  (int(w * 0.15), int(h * 0.15)), 
                  (int(w * 0.85), int(h * 0.85)), 
                  255, -1)
    mask = cv2.bitwise_and(mask, center_mask)
    
    # Dilate to ensure ring is fully covered
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
    
    return mask

def enhance_ring_colors(image: Image.Image, metal_type: str) -> Image.Image:
    """Enhanced color correction for each metal type"""
    # Convert to numpy for processing
    img_array = np.array(image)
    
    # Create ring mask
    ring_mask = detect_ring_mask(img_array)
    
    # Apply enhancements only to ring area
    result = img_array.copy()
    
    # Metal-specific adjustments
    if metal_type == "yellow_gold":
        # Enhance yellow and gold tones
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase brightness
        l_ring = l.copy()
        l_ring[ring_mask > 0] = np.clip(l[ring_mask > 0] * 1.2 + 20, 0, 255)
        
        # Enhance yellow (positive b channel)
        b_ring = b.copy()
        b_ring[ring_mask > 0] = np.clip(b[ring_mask > 0] * 1.3 + 10, 0, 255)
        
        enhanced_lab = cv2.merge([l_ring, a, b_ring])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
    elif metal_type == "rose_gold":
        # Enhance pink/rose tones
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase brightness slightly
        l_ring = l.copy()
        l_ring[ring_mask > 0] = np.clip(l[ring_mask > 0] * 1.15 + 15, 0, 255)
        
        # Enhance pink (positive a channel)
        a_ring = a.copy()
        a_ring[ring_mask > 0] = np.clip(a[ring_mask > 0] * 1.2 + 8, 0, 255)
        
        enhanced_lab = cv2.merge([l_ring, a_ring, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
    elif metal_type in ["white_gold", "white_noplating"]:
        # Enhance brightness and reduce color cast
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase brightness significantly
        l_ring = l.copy()
        l_ring[ring_mask > 0] = np.clip(l[ring_mask > 0] * 1.3 + 30, 0, 255)
        
        # Reduce color cast (neutralize a and b channels)
        a_ring = a.copy()
        b_ring = b.copy()
        a_ring[ring_mask > 0] = np.clip(a[ring_mask > 0] * 0.7 + 32, 0, 255)
        b_ring[ring_mask > 0] = np.clip(b[ring_mask > 0] * 0.7 + 32, 0, 255)
        
        enhanced_lab = cv2.merge([l_ring, a_ring, b_ring])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL
    enhanced = Image.fromarray(result)
    
    # Apply additional PIL enhancements
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(1.15)
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.3)
    
    return enhanced

def detect_metal_type(image: Image.Image) -> str:
    """Detect metal type with improved accuracy"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get center crop for analysis
    width, height = image.size
    crop_size = min(width, height) // 2
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    center_crop = image.crop((left, top, left + crop_size, top + crop_size))
    
    # Analyze colors
    pixels = list(center_crop.getdata())
    
    # Calculate color statistics
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    
    # Calculate color differences
    rg_diff = abs(avg_r - avg_g)
    gb_diff = abs(avg_g - avg_b)
    rb_diff = abs(avg_r - avg_b)
    
    brightness = (avg_r + avg_g + avg_b) / 3
    
    # Detect white metals first (they're most distinctive)
    if brightness > 180 and rg_diff < 15 and gb_diff < 15 and rb_diff < 15:
        # Very bright and neutral = white gold or white no plating
        if brightness > 200:
            return "white_gold"
        else:
            return "white_noplating"
    
    # Rose gold detection (pinkish hue)
    elif avg_r > avg_g > avg_b and rg_diff > 10 and brightness > 140:
        return "rose_gold"
    
    # Default to yellow gold for all other cases
    else:
        return "yellow_gold"

def create_thumbnail_ultra_zoom(original: Image.Image, processed: Image.Image) -> Image.Image:
    """Create thumbnail with wedding ring filling the frame"""
    # Convert to numpy for processing
    img_array = np.array(processed)
    
    # Detect ring area
    ring_mask = detect_ring_mask(img_array)
    
    # Find bounding box of ring
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (main ring area)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add 2% padding
        padding = 0.02
        x = max(0, int(x - w * padding))
        y = max(0, int(y - h * padding))
        w = min(processed.width - x, int(w * (1 + 2 * padding)))
        h = min(processed.height - y, int(h * (1 + 2 * padding)))
        
        # Crop to ring area
        ring_crop = processed.crop((x, y, x + w, y + h))
    else:
        # Fallback: use center 98% crop
        width, height = processed.size
        crop_pct = 0.98
        left = int(width * (1 - crop_pct) / 2)
        top = int(height * (1 - crop_pct) / 2)
        right = int(width * (1 + crop_pct) / 2)
        bottom = int(height * (1 + crop_pct) / 2)
        ring_crop = processed.crop((left, top, right, bottom))
    
    # Create thumbnail with exact dimensions
    thumbnail = Image.new('RGB', (1000, 1300), (248, 248, 248))
    
    # Calculate scaling to fill frame
    scale_w = 1000 / ring_crop.width
    scale_h = 1300 / ring_crop.height
    scale = max(scale_w, scale_h) * 1.02  # 2% extra to ensure full coverage
    
    new_width = int(ring_crop.width * scale)
    new_height = int(ring_crop.height * scale)
    
    # Resize ring
    ring_resized = ring_crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center in frame
    x_offset = (1000 - new_width) // 2
    y_offset = (1300 - new_height) // 2
    
    # Paste ring
    thumbnail.paste(ring_resized, (x_offset, y_offset))
    
    # Enhance brightness slightly for thumbnail
    enhancer = ImageEnhance.Brightness(thumbnail)
    thumbnail = enhancer.enhance(1.05)
    
    return thumbnail

def process_wedding_ring_v99(image_base64: str) -> Dict:
    """Process wedding ring with v99 ultimate precision black removal"""
    try:
        # Decode image
        image_data = base64.b64decode(image_base64)
        original_image = Image.open(BytesIO(image_data))
        
        # Ensure RGB
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(original_image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Ultimate precision black border detection
        print("Detecting black borders with v99 ultimate precision...")
        top, left, bottom, right = detect_black_border_ultimate(image_bgr)
        
        print(f"Black border detection - Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
        
        # Crop image
        if top < bottom and left < right:
            cropped_bgr = image_bgr[top:bottom, left:right]
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(cropped_rgb)
        else:
            processed_image = original_image
        
        # Create clean background
        width, height = processed_image.size
        final_image = Image.new('RGB', (width, height), (248, 248, 248))
        
        # Detect metal type
        metal_type = detect_metal_type(processed_image)
        print(f"Detected metal type: {metal_type}")
        
        # Enhance ring colors
        enhanced_ring = enhance_ring_colors(processed_image, metal_type)
        
        # Paste enhanced ring on clean background
        final_image.paste(enhanced_ring, (0, 0))
        
        # Create thumbnail
        thumbnail = create_thumbnail_ultra_zoom(original_image, final_image)
        
        # Convert to base64
        # Main image
        main_buffer = BytesIO()
        final_image.save(main_buffer, format='JPEG', quality=95, optimize=True)
        main_buffer.seek(0)
        main_base64 = base64.b64encode(main_buffer.read()).decode('utf-8')
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='JPEG', quality=95, optimize=True)
        thumb_buffer.seek(0)
        thumb_base64 = base64.b64encode(thumb_buffer.read()).decode('utf-8')
        
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "metal_type": metal_type,
                "processing_version": "v99_ultimate_precision",
                "border_removed": {
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v99: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v99_ultimate_precision"
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        # Get input
        input_data = event.get("input", {})
        
        # Check for test mode
        if input_data.get("test") == True:
            return {
                "status": "test_success",
                "message": "Wedding Ring Processor v99 - Ultimate Precision Ready",
                "version": "v99_ultimate_precision",
                "features": [
                    "Connected Component Analysis",
                    "Line Continuity Detection",
                    "Diagonal Corner Scanning",
                    "Multi-threshold Progressive Scan",
                    "Multi-color Space Detection (RGB+HSV+LAB)",
                    "Morphological Operations",
                    "Ultimate Precision Black Removal"
                ]
            }
        
        # Get image
        image_base64 = input_data.get("image")
        if not image_base64:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Process image
        return process_wedding_ring_v99(image_base64)
        
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
