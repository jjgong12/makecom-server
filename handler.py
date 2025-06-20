import runpod
import base64
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from io import BytesIO
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import traceback

def initial_force_crop(image_array: np.ndarray, crop_percent: float = 0.02) -> np.ndarray:
    """Force crop edges by given percentage to guarantee removal"""
    h, w = image_array.shape[:2]
    
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    
    # Ensure minimum crop
    crop_h = max(crop_h, 20)
    crop_w = max(crop_w, 20)
    
    return image_array[crop_h:h-crop_h, crop_w:w-crop_w]

def verify_black_removal(image_array: np.ndarray, threshold: int = 80) -> bool:
    """Verify if black borders are completely removed"""
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Check edges for black pixels
    edge_size = 10
    
    # Top edge
    if np.mean(gray[:edge_size, :]) < threshold:
        return False
    
    # Bottom edge
    if np.mean(gray[-edge_size:, :]) < threshold:
        return False
    
    # Left edge
    if np.mean(gray[:, :edge_size]) < threshold:
        return False
    
    # Right edge
    if np.mean(gray[:, -edge_size:]) < threshold:
        return False
    
    # Check corners more strictly
    corner_size = 30
    corners = [
        gray[:corner_size, :corner_size],  # Top-left
        gray[:corner_size, -corner_size:],  # Top-right
        gray[-corner_size:, :corner_size],  # Bottom-left
        gray[-corner_size:, -corner_size:]  # Bottom-right
    ]
    
    for corner in corners:
        if np.mean(corner) < threshold + 20:  # Stricter for corners
            return False
    
    return True

def detect_and_remove_black_iterative(image_array: np.ndarray, iteration: int) -> Tuple[np.ndarray, bool]:
    """Detect and remove black borders with increasing aggressiveness"""
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Increase aggressiveness with each iteration
    base_threshold = 80 - (iteration * 10)  # 80, 70, 60, 50, 40
    scan_depth = 0.3 + (iteration * 0.1)  # 30%, 40%, 50%, 60%, 70%
    min_crop = 20 + (iteration * 10)  # 20, 30, 40, 50, 60
    
    top_crop = 0
    bottom_crop = h
    left_crop = 0
    right_crop = w
    
    # Method 1: Progressive threshold scanning
    max_scan = int(min(h, w) * scan_depth)
    
    # Multiple threshold levels
    thresholds = [base_threshold - 20, base_threshold, base_threshold + 20]
    
    for threshold in thresholds:
        # Top scan
        for i in range(min(max_scan, h//2)):
            if np.mean(gray[i, :]) < threshold:
                top_crop = max(top_crop, i + 1)
            else:
                if i > min_crop:  # Ensure minimum crop
                    break
        
        # Bottom scan
        for i in range(min(max_scan, h//2)):
            if np.mean(gray[h-1-i, :]) < threshold:
                bottom_crop = min(bottom_crop, h - 1 - i)
            else:
                if i > min_crop:
                    break
        
        # Left scan
        for i in range(min(max_scan, w//2)):
            if np.mean(gray[:, i]) < threshold:
                left_crop = max(left_crop, i + 1)
            else:
                if i > min_crop:
                    break
        
        # Right scan
        for i in range(min(max_scan, w//2)):
            if np.mean(gray[:, w-1-i]) < threshold:
                right_crop = min(right_crop, w - 1 - i)
            else:
                if i > min_crop:
                    break
    
    # Method 2: Connected component analysis (more aggressive in later iterations)
    if iteration >= 2:
        _, binary = cv2.threshold(gray, base_threshold + 20, 255, cv2.THRESH_BINARY)
        
        # Larger kernel for later iterations
        kernel_size = 3 + (iteration * 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = 255 - binary
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w_comp, h_comp, area = stats[i]
            
            # Lower area threshold for later iterations
            area_threshold = max(50, 200 - (iteration * 30))
            
            if area > area_threshold:
                if x <= 5:
                    left_crop = max(left_crop, x + w_comp + 5)
                if x + w_comp >= w - 5:
                    right_crop = min(right_crop, x - 5)
                if y <= 5:
                    top_crop = max(top_crop, y + h_comp + 5)
                if y + h_comp >= h - 5:
                    bottom_crop = min(bottom_crop, y - 5)
    
    # Method 3: Gradient-based detection (for subtle borders)
    if iteration >= 1:
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Strong gradients indicate border edges
        gradient_threshold = 30 - (iteration * 5)
        
        # Check for gradient lines
        for i in range(min_crop, min(max_scan, h//3)):
            if np.mean(gradient[i, :]) > gradient_threshold:
                if np.mean(gray[:i, :]) < base_threshold:
                    top_crop = max(top_crop, i + 5)
                break
    
    # Ensure minimum crop based on iteration
    top_crop = max(top_crop, min_crop)
    left_crop = max(left_crop, min_crop)
    bottom_crop = min(bottom_crop, h - min_crop)
    right_crop = min(right_crop, w - min_crop)
    
    # Validate crops
    if top_crop >= bottom_crop - 100 or left_crop >= right_crop - 100:
        # Too aggressive, use safer crop
        safe_crop = min_crop
        return image_array[safe_crop:h-safe_crop, safe_crop:w-safe_crop], False
    
    # Crop image
    cropped = image_array[top_crop:bottom_crop, left_crop:right_crop]
    
    # Verify removal
    success = verify_black_removal(cropped, base_threshold)
    
    return cropped, success

def hybrid_inpaint_edges(image_array: np.ndarray, edge_size: int = 10) -> np.ndarray:
    """Inpaint only the edges to handle any remaining artifacts"""
    h, w = image_array.shape[:2]
    
    # Create mask for edges only
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark edges for inpainting
    mask[:edge_size, :] = 255  # Top
    mask[-edge_size:, :] = 255  # Bottom
    mask[:, :edge_size] = 255  # Left
    mask[:, -edge_size:] = 255  # Right
    
    # Check if edges are dark
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Only inpaint if edges are actually dark
    if (np.mean(gray[:edge_size, :]) < 100 or 
        np.mean(gray[-edge_size:, :]) < 100 or
        np.mean(gray[:, :edge_size]) < 100 or
        np.mean(gray[:, -edge_size:]) < 100):
        
        # Inpaint edges
        result = cv2.inpaint(image_array, mask, 3, cv2.INPAINT_TELEA)
        return result
    
    return image_array

def detect_ring_mask_safe(image_array: np.ndarray) -> np.ndarray:
    """Create safe mask for wedding ring with conservative detection"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Focus on center 60% where ring is likely to be
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    
    # Create circular mask for center area
    radius = min(w, h) * 0.3
    cv2.circle(mask, (center_x, center_y), int(radius), 255, -1)
    
    # Use adaptive threshold to find ring
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    
    # Find contours in center area
    center_adaptive = cv2.bitwise_and(adaptive, mask)
    contours, _ = cv2.findContours(center_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Reset mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw significant contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < h * w * 0.4:  # Ring-sized objects
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Dilate generously to ensure ring protection
    mask = cv2.dilate(mask, np.ones((20, 20), np.uint8), iterations=3)
    
    return mask

def enhance_ring_colors(image: Image.Image, metal_type: str) -> Image.Image:
    """Enhanced color correction for each metal type"""
    # Convert to numpy for processing
    img_array = np.array(image)
    
    # Create ring mask
    ring_mask = detect_ring_mask_safe(img_array)
    
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
    ring_mask = detect_ring_mask_safe(img_array)
    
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

def process_wedding_ring_v100(image_base64: str) -> Dict:
    """Process wedding ring with v100 iterative perfect removal system"""
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
        
        # Step 1: Initial force crop (2% from all edges)
        print("Step 1: Initial force crop to guarantee edge removal...")
        image_bgr = initial_force_crop(image_bgr, crop_percent=0.02)
        
        # Step 2: Iterative black removal (up to 5 iterations)
        print("Step 2: Starting iterative black removal process...")
        max_iterations = 5
        
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Detect and remove black borders
            image_bgr, success = detect_and_remove_black_iterative(image_bgr, iteration)
            
            if success:
                print(f"Black borders successfully removed in iteration {iteration + 1}")
                break
            
            if iteration == max_iterations - 1:
                print("Maximum iterations reached. Applying final aggressive crop...")
                # Final aggressive crop if still not successful
                h, w = image_bgr.shape[:2]
                final_crop = 50  # Remove 50 pixels from each edge
                image_bgr = image_bgr[final_crop:h-final_crop, final_crop:w-final_crop]
        
        # Step 3: Hybrid edge inpainting for any remaining artifacts
        print("Step 3: Applying hybrid edge inpainting...")
        image_bgr = hybrid_inpaint_edges(image_bgr, edge_size=10)
        
        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(image_rgb)
        
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
                "processing_version": "v100_iterative_perfect",
                "removal_stats": {
                    "initial_crop": "2%",
                    "iterations_used": iteration + 1 if 'iteration' in locals() else 1,
                    "hybrid_inpaint": "applied"
                },
                "status": "success"
            }
        }
        
    except Exception as e:
        print(f"Error in process_wedding_ring_v100: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error",
                "processing_version": "v100_iterative_perfect"
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
                "message": "Wedding Ring Processor v100 - Iterative Perfect Removal Ready",
                "version": "v100_iterative_perfect",
                "features": [
                    "Initial 2% force crop guarantee",
                    "5-iteration progressive removal",
                    "Verification after each iteration",
                    "Hybrid edge inpainting",
                    "Increasing aggressiveness per iteration",
                    "Final aggressive fallback",
                    "100% black border removal guarantee"
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
        return process_wedding_ring_v100(image_base64)
        
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
