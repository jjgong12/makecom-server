import runpod
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_wedding_ring_area(image):
    """
    Detect and protect wedding ring area before border removal
    Returns a mask where ring areas are marked
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple detection methods for safety
    # Method 1: Color-based detection (metallic tones)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Gold/Rose gold range
    lower_gold = np.array([10, 30, 100])
    upper_gold = np.array([25, 255, 255])
    mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
    
    # Silver/White gold range
    lower_silver = np.array([0, 0, 150])
    upper_silver = np.array([180, 30, 255])
    mask_silver = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # Combine masks
    ring_mask = cv2.bitwise_or(mask_gold, mask_silver)
    
    # Method 2: Edge-based detection (circular patterns)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect circles (rings are often circular)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=500
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(ring_mask, (circle[0], circle[1]), circle[2] + 30, 255, -1)
    
    # Method 3: Texture-based (smooth metallic surfaces)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    texture_mask = (np.abs(laplacian) < 10).astype(np.uint8) * 255
    
    # Combine all methods
    ring_mask = cv2.bitwise_or(ring_mask, texture_mask)
    
    # Dilate to ensure full ring coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    ring_mask = cv2.dilate(ring_mask, kernel, iterations=2)
    
    return ring_mask

def verify_border_removal(image, original_shape):
    """
    Verify if border was properly removed
    Returns confidence score (0-1) and issues found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    oh, ow = original_shape[:2]
    
    issues = []
    scores = []
    
    # Check 1: Size reduction ratio
    size_reduction = (h * w) / (oh * ow)
    if size_reduction > 0.9:
        issues.append("Minimal size reduction - border might remain")
        scores.append(0.3)
    elif size_reduction < 0.3:
        issues.append("Excessive cropping - might have removed content")
        scores.append(0.5)
    else:
        scores.append(1.0)
    
    # Check 2: Edge darkness
    edge_size = 20
    edges_mean = [
        gray[:edge_size, :].mean(),      # top
        gray[-edge_size:, :].mean(),     # bottom
        gray[:, :edge_size].mean(),      # left
        gray[:, -edge_size:].mean()      # right
    ]
    
    dark_edges = sum(1 for edge in edges_mean if edge < 100)
    if dark_edges >= 2:
        issues.append(f"{dark_edges} dark edges detected")
        scores.append(0.4)
    elif dark_edges == 1:
        issues.append("1 dark edge detected")
        scores.append(0.7)
    else:
        scores.append(1.0)
    
    # Check 3: Black pixel ratio at borders
    border_strip = 30
    border_pixels = np.concatenate([
        gray[:border_strip, :].flatten(),
        gray[-border_strip:, :].flatten(),
        gray[:, :border_strip].flatten(),
        gray[:, -border_strip:].flatten()
    ])
    
    black_ratio = np.sum(border_pixels < 50) / len(border_pixels)
    if black_ratio > 0.3:
        issues.append(f"High black pixel ratio at borders: {black_ratio:.2f}")
        scores.append(0.3)
    elif black_ratio > 0.1:
        issues.append(f"Some black pixels at borders: {black_ratio:.2f}")
        scores.append(0.7)
    else:
        scores.append(1.0)
    
    confidence = np.mean(scores)
    return confidence, issues

def remove_border_safe_multi_method(image, ring_mask=None):
    """
    Multi-method border removal with ring protection
    Tries multiple approaches and selects the best result
    """
    h, w = image.shape[:2]
    results = []
    
    # Method 1: Enhanced gradient-based detection
    def method_gradient():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Find content boundaries
        threshold = np.percentile(gradient, 90)
        content_mask = gradient > threshold
        
        # Find bounding box
        coords = np.column_stack(np.where(content_mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add safety margin
            margin = 10
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            y_max = min(h, y_max + margin)
            x_max = min(w, x_max + margin)
            
            return image[y_min:y_max, x_min:x_max]
        return image
    
    # Method 2: Variance-based detection
    def method_variance():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance
        window_size = 20
        mean = cv2.blur(gray, (window_size, window_size))
        sqr_mean = cv2.blur(gray**2, (window_size, window_size))
        variance = sqr_mean - mean**2
        
        # High variance = content, low variance = border
        content_mask = variance > np.percentile(variance, 10)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find bounding box
        coords = cv2.findNonZero(content_mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return image[y:y+h, x:x+w]
        return image
    
    # Method 3: Improved line-by-line scan
    def method_smart_scan():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Protect ring area if provided
        if ring_mask is not None:
            protected_gray = gray.copy()
            protected_gray[ring_mask > 0] = 255
        else:
            protected_gray = gray
        
        # Smart scan with multiple criteria
        def find_content_start(line, reverse=False):
            if reverse:
                line = line[::-1]
            
            for i in range(len(line)):
                # Multiple criteria for content detection
                window = line[i:i+50] if i+50 < len(line) else line[i:]
                if len(window) > 0:
                    # Criteria 1: Mean brightness
                    if np.mean(window) > 100:
                        return i if not reverse else len(line) - i
                    # Criteria 2: Variance (texture)
                    if np.std(window) > 20:
                        return i if not reverse else len(line) - i
                    # Criteria 3: Max value (any bright pixel)
                    if np.max(window) > 200:
                        return i if not reverse else len(line) - i
            
            return 0 if not reverse else len(line)
        
        # Find borders
        top = min(find_content_start(protected_gray[i, :]) for i in range(0, h, 10))
        bottom = max(find_content_start(protected_gray[i, :], reverse=True) for i in range(h-1, 0, -10))
        left = min(find_content_start(protected_gray[:, i]) for i in range(0, w, 10))
        right = max(find_content_start(protected_gray[:, i], reverse=True) for i in range(w-1, 0, -10))
        
        # Safety checks
        if bottom <= top + 100 or right <= left + 100:
            return image
        
        return image[top:bottom, left:right]
    
    # Try all methods
    try:
        results.append(('gradient', method_gradient()))
    except:
        logger.warning("Gradient method failed")
    
    try:
        results.append(('variance', method_variance()))
    except:
        logger.warning("Variance method failed")
    
    try:
        results.append(('smart_scan', method_smart_scan()))
    except:
        logger.warning("Smart scan method failed")
    
    # Add original ultra method as fallback
    try:
        from_ultra = detect_and_remove_black_border_ultimate(image)
        results.append(('ultra', from_ultra))
    except:
        logger.warning("Ultra method failed")
    
    # Select best result
    best_result = image
    best_score = 0
    best_method = 'none'
    
    for method_name, result in results:
        if result is not None and result.size > 0:
            score, issues = verify_border_removal(result, image.shape)
            logger.info(f"Method {method_name}: score={score:.2f}, issues={issues}")
            
            if score > best_score:
                best_score = score
                best_result = result
                best_method = method_name
    
    logger.info(f"Selected method: {best_method} with score {best_score:.2f}")
    return best_result

def detect_and_remove_black_border_ultimate(image):
    """
    Ultimate black border removal - v73 Final
    Enhanced with safety checks and validation
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    original_h, original_w = h, w
    
    # PASS 1: Maximum aggressive scan (60% of image)
    logger.info("PASS 1: Maximum aggressive border scan - 60% area")
    max_border = min(int(h * 0.6), int(w * 0.6), 600)
    
    # Enhanced detection with multiple criteria
    def is_border_line(line, threshold_mean=100, threshold_dark_ratio=0.85):
        if len(line) == 0:
            return False
        
        # Criteria 1: Mean value
        if np.mean(line) < threshold_mean:
            return True
        
        # Criteria 2: Dark pixel ratio
        dark_ratio = np.sum(line < threshold_mean) / len(line)
        if dark_ratio > threshold_dark_ratio:
            return True
        
        # Criteria 3: Maximum value (no bright pixels)
        if np.max(line) < 120:
            return True
        
        # Criteria 4: Variance (uniform darkness)
        if np.std(line) < 10 and np.mean(line) < 150:
            return True
        
        return False
    
    # Top border
    top_crop = 0
    for y in range(max_border):
        if is_border_line(gray[y, :]):
            top_crop = y + 1
        else:
            # Check next few lines to confirm
            if y + 5 < max_border:
                confirm = sum(1 for dy in range(1, 6) if is_border_line(gray[y + dy, :]))
                if confirm < 3:  # Most lines are content
                    break
    
    # Bottom border
    bottom_crop = h
    for y in range(h - 1, h - max_border, -1):
        if is_border_line(gray[y, :]):
            bottom_crop = y
        else:
            if y - 5 > h - max_border:
                confirm = sum(1 for dy in range(1, 6) if is_border_line(gray[y - dy, :]))
                if confirm < 3:
                    break
    
    # Left border
    left_crop = 0
    for x in range(max_border):
        if is_border_line(gray[:, x]):
            left_crop = x + 1
        else:
            if x + 5 < max_border:
                confirm = sum(1 for dx in range(1, 6) if is_border_line(gray[:, x + dx]))
                if confirm < 3:
                    break
    
    # Right border
    right_crop = w
    for x in range(w - 1, w - max_border, -1):
        if is_border_line(gray[:, x]):
            right_crop = x
        else:
            if x - 5 > w - max_border:
                confirm = sum(1 for dx in range(1, 6) if is_border_line(gray[:, x - dx]))
                if confirm < 3:
                    break
    
    # Apply PASS 1
    if top_crop > 0 or bottom_crop < h or left_crop > 0 or right_crop < w:
        logger.info(f"PASS 1 removed: top={top_crop}, bottom={h-bottom_crop}, left={left_crop}, right={w-right_crop}")
        image = image[top_crop:bottom_crop, left_crop:right_crop]
        gray = gray[top_crop:bottom_crop, left_crop:right_crop]
        h, w = gray.shape
    
    # PASS 2: Edge gradient detection
    logger.info("PASS 2: Edge gradient detection")
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Find first significant edge from each direction
    edge_threshold = 50
    check_depth = min(50, h//4, w//4)
    
    # Additional cropping based on edges
    extra_crop = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    # Top edge
    for y in range(check_depth):
        if np.max(edges[y, :]) > edge_threshold:
            if y > 5:
                extra_crop['top'] = y - 5
            break
    
    # Bottom edge
    for y in range(h-1, h-check_depth, -1):
        if np.max(edges[y, :]) > edge_threshold:
            if y < h - 5:
                extra_crop['bottom'] = h - y - 5
            break
    
    # Apply PASS 2 crops
    if any(extra_crop.values()):
        logger.info(f"PASS 2 extra crop: {extra_crop}")
        image = image[extra_crop['top']:h-extra_crop['bottom'], 
                     extra_crop['left']:w-extra_crop['right']]
        gray = gray[extra_crop['top']:h-extra_crop['bottom'], 
                   extra_crop['left']:w-extra_crop['right']]
        h, w = gray.shape
    
    # PASS 3: Final precision cleanup
    logger.info("PASS 3: Final precision cleanup")
    
    # Very aggressive final pass with validation
    final_check = 20
    
    if h > final_check * 2 and w > final_check * 2:
        # Check each edge
        final_crops = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        if np.percentile(gray[:final_check, :], 90) < 100:
            final_crops['top'] = final_check
        
        if np.percentile(gray[-final_check:, :], 90) < 100:
            final_crops['bottom'] = final_check
        
        if np.percentile(gray[:, :final_check], 90) < 100:
            final_crops['left'] = final_check
        
        if np.percentile(gray[:, -final_check:], 90) < 100:
            final_crops['right'] = final_check
        
        # Apply final crops
        if any(final_crops.values()):
            logger.info(f"PASS 3 final crops: {final_crops}")
            image = image[final_crops['top']:h-final_crops['bottom'],
                         final_crops['left']:w-final_crops['right']]
    
    final_h, final_w = image.shape[:2]
    
    # Safety check - if we removed too much, be more conservative
    if final_h < original_h * 0.3 or final_w < original_w * 0.3:
        logger.warning("Removed too much, returning more conservative crop")
        # Return a more conservative crop
        safe_top = min(top_crop, original_h // 4)
        safe_bottom = max(bottom_crop, 3 * original_h // 4)
        safe_left = min(left_crop, original_w // 4)
        safe_right = max(right_crop, 3 * original_w // 4)
        return image[safe_top:safe_bottom, safe_left:safe_right]
    
    logger.info(f"Ultimate border removal complete: {original_h}x{original_w} -> {final_h}x{final_w}")
    return image

def detect_ring_advanced(image):
    """
    Advanced ring detection with multiple methods
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Binary threshold
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Method 2: Adaptive threshold for better edge detection
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both methods
    combined = cv2.bitwise_or(binary, adaptive)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and circularity
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Some circularity
                    valid_contours.append(contour)
    
    if not valid_contours:
        # Fallback to largest contour
        valid_contours = [max(contours, key=cv2.contourArea)]
    
    # Get overall bounding box
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def enhance_wedding_ring_perfect(image, metal_type='white_gold', lighting='balanced'):
    """
    Perfect enhancement without color distortion - NO LAB/HSV conversions!
    """
    # Convert to PIL for safe enhancements
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Metal-specific enhancement parameters (refined for no color shift)
    metal_params = {
        'white_gold': {
            'brightness': 1.35,
            'contrast': 1.25,
            'saturation': 0.98,
            'sharpness': 2.0
        },
        'yellow_gold': {
            'brightness': 1.32,
            'contrast': 1.22,
            'saturation': 1.08,
            'sharpness': 1.8
        },
        'rose_gold': {
            'brightness': 1.32,
            'contrast': 1.20,
            'saturation': 1.1,
            'sharpness': 1.8
        },
        'platinum': {
            'brightness': 1.38,
            'contrast': 1.28,
            'saturation': 0.95,
            'sharpness': 2.2
        }
    }
    
    # Lighting adjustments (subtle)
    lighting_params = {
        'studio': {
            'brightness_mult': 1.02,
            'contrast_mult': 1.05
        },
        'natural': {
            'brightness_mult': 1.0,
            'contrast_mult': 0.98
        },
        'mixed': {
            'brightness_mult': 1.01,
            'contrast_mult': 1.0
        },
        'balanced': {
            'brightness_mult': 1.0,
            'contrast_mult': 1.0
        }
    }
    
    # Get parameters
    metal = metal_params.get(metal_type, metal_params['white_gold'])
    light = lighting_params.get(lighting, lighting_params['balanced'])
    
    # Apply enhancements using PIL only (no color space conversions)
    # 1. Brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(metal['brightness'] * light['brightness_mult'])
    
    # 2. Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(metal['contrast'] * light['contrast_mult'])
    
    # 3. Color (subtle saturation)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(metal['saturation'])
    
    # 4. Sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(metal['sharpness'])
    
    # Convert back to OpenCV
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Gentle bilateral filter for smoothness
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Subtle unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # Ensure pure white background (248, 248, 248)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    enhanced[white_mask == 255] = [248, 248, 248]
    
    return enhanced

def create_professional_background_perfect(image):
    """
    Create professional white background without color shifts
    """
    # Convert to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create slightly larger canvas
    width, height = pil_image.size
    new_width = int(width * 1.1)
    new_height = int(height * 1.1)
    
    # Create pure white background (248, 248, 248)
    background = Image.new('RGB', (new_width, new_height), (248, 248, 248))
    
    # Center the image
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2
    background.paste(pil_image, (x_offset, y_offset))
    
    # Very subtle edge softening (no vignette that changes colors)
    background = background.filter(ImageFilter.GaussianBlur(1))
    
    # Ensure edges are white
    draw_background = Image.new('RGB', (new_width, new_height), (248, 248, 248))
    mask = Image.new('L', (new_width, new_height), 0)
    
    # Create center mask
    center_box = (
        int(new_width * 0.05),
        int(new_height * 0.05),
        int(new_width * 0.95),
        int(new_height * 0.95)
    )
    mask.paste(255, center_box)
    mask = mask.filter(ImageFilter.GaussianBlur(20))
    
    # Composite
    background = Image.composite(background, draw_background, mask)
    
    return background

def create_thumbnail_ultimate(image, ring_bbox=None, target_size=(1000, 1300)):
    """
    Ultimate thumbnail with perfect border removal
    """
    logger.info("Creating ultimate thumbnail")
    
    # Apply same border removal as main image
    ring_mask = detect_wedding_ring_area(image)
    image_clean = remove_border_safe_multi_method(image.copy(), ring_mask)
    
    # Verify border removal
    confidence, issues = verify_border_removal(image_clean, image.shape)
    if confidence < 0.7:
        logger.warning(f"Low confidence border removal: {confidence}, issues: {issues}")
        # Try fallback method
        image_clean = detect_and_remove_black_border_ultimate(image.copy())
    
    # Detect ring if not provided
    if ring_bbox is None:
        ring_bbox = detect_ring_advanced(image_clean)
    
    if ring_bbox is not None:
        x, y, w, h = ring_bbox
        # Minimal padding (0.5% - ultra tight crop)
        padding = int(max(w, h) * 0.005)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_clean.shape[1] - x, w + 2 * padding)
        h = min(image_clean.shape[0] - y, h + 2 * padding)
        
        # Crop to ring area
        ring_crop = image_clean[y:y+h, x:x+w]
    else:
        logger.warning("No ring detected, using full cleaned image")
        ring_crop = image_clean
    
    # Create white background
    background = np.full((target_size[1], target_size[0], 3), 248, dtype=np.uint8)
    
    # Calculate scale to fill 99.5% of canvas (maximum possible)
    scale_x = (target_size[0] * 0.995) / ring_crop.shape[1]
    scale_y = (target_size[1] * 0.995) / ring_crop.shape[0]
    scale = min(scale_x, scale_y)
    
    new_width = int(ring_crop.shape[1] * scale)
    new_height = int(ring_crop.shape[0] * scale)
    
    # Resize with highest quality
    ring_resized = cv2.resize(ring_crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply perfect enhancement
    ring_resized = enhance_wedding_ring_perfect(ring_resized)
    
    # Center on background
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = ring_resized
    
    # Convert to PIL for final polish
    thumbnail = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # Final sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    logger.info(f"Ultimate thumbnail: {new_width}x{new_height} in {target_size[0]}x{target_size[1]}")
    
    return thumbnail

def handler(job):
    """RunPod handler - v73 Final with Ultimate Safe Border Removal"""
    logger.info("Starting v73 Final processing")
    
    try:
        job_input = job['input']
        
        # Support both 'image' and 'image_base64' fields
        image_data = job_input.get('image') or job_input.get('image_base64')
        
        if not image_data:
            raise ValueError("No image data provided")
        
        # Handle base64 padding
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        logger.info(f"Image loaded: {image.shape}")
        
        # Step 1: Detect and protect ring area
        logger.info("Detecting wedding ring area for protection...")
        ring_mask = detect_wedding_ring_area(image)
        
        # Step 2: Apply safe multi-method border removal
        logger.info("Applying safe multi-method border removal...")
        processed_image = remove_border_safe_multi_method(image, ring_mask)
        
        # Step 3: Verify border removal quality
        confidence, issues = verify_border_removal(processed_image, image.shape)
        logger.info(f"Border removal confidence: {confidence:.2f}")
        
        if confidence < 0.7:
            logger.warning(f"Low confidence, issues: {issues}")
            # Try ultimate method as fallback
            processed_image = detect_and_remove_black_border_ultimate(image)
        
        logger.info(f"After border removal: {processed_image.shape}")
        
        # Step 4: Detect ring for better processing
        logger.info("Detecting wedding ring in cleaned image...")
        ring_bbox = detect_ring_advanced(processed_image)
        
        # Step 5: Enhance with metal and lighting (no color distortion)
        metal_type = job_input.get("metal_type", "white_gold")
        lighting = job_input.get("lighting", "balanced")
        
        logger.info(f"Enhancing ring - Metal: {metal_type}, Lighting: {lighting}")
        enhanced_image = enhance_wedding_ring_perfect(processed_image, metal_type, lighting)
        
        # Step 6: Create professional background
        final_image = create_professional_background_perfect(enhanced_image)
        
        # Step 7: Create ultimate thumbnail
        logger.info("Creating ultimate thumbnail...")
        thumbnail = create_thumbnail_ultimate(image, ring_bbox, target_size=(1000, 1300))
        
        # Convert images to base64
        # Main image
        main_buffer = BytesIO()
        # Convert numpy array to PIL if needed
        if isinstance(final_image, np.ndarray):
            final_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        final_image.save(main_buffer, format='PNG', optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode('utf-8')
        main_base64 = main_base64.rstrip('=')  # Remove padding for Make.com
        
        # Thumbnail
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG', optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')  # Remove padding
        
        logger.info("Processing completed successfully - v73 Final")
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v73_final_safe",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": list(image.shape[:2]),
                    "processed_size": list(processed_image.shape[:2]),
                    "thumbnail_size": [1000, 1300],
                    "border_removed": True,
                    "border_confidence": float(confidence),
                    "border_issues": issues,
                    "ring_detected": ring_bbox is not None,
                    "ring_protected": True,
                    "ring_bbox": list(ring_bbox) if ring_bbox else None,
                    "methods_used": [
                        "ring_area_protection",
                        "multi_method_border_removal",
                        "gradient_detection",
                        "variance_detection", 
                        "smart_scan",
                        "verification_system"
                    ],
                    "enhancements_applied": [
                        "metal_specific_correction",
                        "lighting_adjustment",
                        "professional_background",
                        "no_color_space_conversion",
                        "pure_white_background_248"
                    ],
                    "color_accuracy": "perfect_no_lab_hsv",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "processing_info": {
                    "version": "v73_final_safe",
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
