import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
import json
import os
import requests
import time

# Replicate API configuration
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '')

def log_debug(message):
    """Debug logging"""
    print(f"[DEBUG] {message}")

def decode_base64_image(base64_string):
    """Decode base64 image with comprehensive error handling"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Handle potential padding issues
        base64_string = base64_string.strip()
        
        # Add padding if needed
        padding = 4 - (len(base64_string) % 4)
        if padding and padding != 4:
            base64_string += '=' * padding
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log_debug(f"Error decoding base64: {str(e)}")
        raise

def encode_image_to_base64(image, format='JPEG'):
    """Encode image to base64 without padding for Make.com"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=95)
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_string.rstrip('=')  # Remove padding for Make.com

def detect_metal_type_in_region(image, bbox=None):
    """Detect metal type within specific region"""
    if bbox:
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w]
    else:
        region = image
    
    # Color analysis in HSV
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Average values
    avg_saturation = np.mean(s)
    avg_brightness = np.mean(v)
    
    # RGB analysis
    b, g, r = cv2.split(region)
    avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)
    
    # Metal type detection with conservative approach
    if avg_saturation < 15 and avg_brightness > 180:
        return "white_gold"
    elif avg_r > avg_g * 1.15 and avg_r > avg_b * 1.2 and avg_saturation > 20:
        return "rose_gold"
    elif avg_saturation < 20 and avg_brightness > 150:
        return "white_gold"
    else:
        # Default to yellow_gold only if clearly yellow
        yellow_ratio = (avg_r + avg_g) / (2 * avg_b) if avg_b > 0 else 1
        if yellow_ratio > 1.3 and avg_saturation > 25:
            return "yellow_gold"
        else:
            return "white_gold"  # Conservative default

def detect_masking_comprehensive(image):
    """Enhanced masking detection including center-based detection"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    log_debug(f"Image size: {w}x{h}")
    
    best_mask = None
    best_score = 0
    
    # Strategy 1: Edge-based progressive scanning (existing)
    edge_masks = detect_edge_based_masking(gray, w, h)
    
    # Strategy 2: Center-based detection (NEW)
    center_masks = detect_center_based_masking(gray, w, h)
    
    # Strategy 3: Contour-based detection (NEW)
    contour_masks = detect_contour_based_masking(gray, w, h)
    
    # Combine all detected masks
    all_masks = edge_masks + center_masks + contour_masks
    
    # Evaluate and choose best mask
    for mask in all_masks:
        if validate_rectangular_mask(mask, w, h):
            score = calculate_mask_score(mask, gray)
            if score > best_score:
                best_score = score
                best_mask = mask
    
    if best_mask:
        log_debug(f"Best mask found: {best_mask['type']} with score {best_score}")
        return best_mask
    
    return None

def detect_edge_based_masking(gray, w, h):
    """Original edge-based detection"""
    masks = []
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    threshold_values = [20, 30, 40, 50, 60, 70, 80]
    
    for scan_pct, threshold in zip(scan_percentages, threshold_values):
        scan_depth = int(min(w, h) * scan_pct)
        
        # Check each edge
        edges_with_black = []
        
        # Top edge
        if np.mean(gray[:scan_depth, :] < threshold) > 0.8:
            edges_with_black.append('top')
        
        # Bottom edge
        if np.mean(gray[-scan_depth:, :] < threshold) > 0.8:
            edges_with_black.append('bottom')
        
        # Left edge
        if np.mean(gray[:, :scan_depth] < threshold) > 0.8:
            edges_with_black.append('left')
        
        # Right edge
        if np.mean(gray[:, -scan_depth:] < threshold) > 0.8:
            edges_with_black.append('right')
        
        if len(edges_with_black) >= 2:
            # Determine bounds
            bounds = calculate_bounds_from_edges(gray, edges_with_black, scan_depth, threshold)
            if bounds:
                masks.append({
                    'bounds': bounds,
                    'type': 'edge_based',
                    'edges': edges_with_black
                })
    
    return masks

def detect_center_based_masking(gray, w, h):
    """New center-based detection for masks not touching edges"""
    masks = []
    
    # Focus on central 80% of image
    center_x1 = int(w * 0.1)
    center_x2 = int(w * 0.9)
    center_y1 = int(h * 0.1)
    center_y2 = int(h * 0.9)
    
    center_region = gray[center_y1:center_y2, center_x1:center_x2]
    
    # Multiple thresholds
    for threshold in [30, 40, 50]:
        _, black_mask = cv2.threshold(center_region, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Must be at least 5% of image area
            if area > (w * h * 0.05):
                # Check if rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:  # Rectangular
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # Adjust coordinates back to full image
                    x += center_x1
                    y += center_y1
                    
                    # Account for line thickness (15-20px)
                    line_thickness = 20
                    inner_x = x + line_thickness
                    inner_y = y + line_thickness
                    inner_w = w_box - 2 * line_thickness
                    inner_h = h_box - 2 * line_thickness
                    
                    if inner_w > 0 and inner_h > 0:
                        masks.append({
                            'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                            'type': 'center_rectangle',
                            'thickness': line_thickness
                        })
    
    return masks

def detect_contour_based_masking(gray, w, h):
    """Contour-based detection for any rectangular masks"""
    masks = []
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (w * h * 0.03):  # At least 3% of image
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Rectangular
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Calculate inner bounds
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Check if it's mostly black inside
                mask_region = gray[y:y+h_box, x:x+w_box]
                if np.mean(mask_region < 50) > 0.7:  # 70% black pixels
                    line_thickness = estimate_line_thickness(gray, x, y, w_box, h_box)
                    
                    inner_x = x + line_thickness
                    inner_y = y + line_thickness
                    inner_w = w_box - 2 * line_thickness
                    inner_h = h_box - 2 * line_thickness
                    
                    if inner_w > 0 and inner_h > 0:
                        masks.append({
                            'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                            'type': 'contour_based',
                            'thickness': line_thickness
                        })
    
    return masks

def estimate_line_thickness(gray, x, y, w, h):
    """Estimate the thickness of black lines"""
    thickness_samples = []
    
    # Sample at multiple points
    for offset in [0.25, 0.5, 0.75]:
        # Top edge
        sample_x = int(x + w * offset)
        thickness = 0
        for scan_y in range(y, max(0, y-50), -1):
            if scan_y < gray.shape[0] and sample_x < gray.shape[1]:
                if gray[scan_y, sample_x] < 40:
                    thickness += 1
                else:
                    break
        if thickness > 0:
            thickness_samples.append(thickness)
        
        # Left edge
        sample_y = int(y + h * offset)
        thickness = 0
        for scan_x in range(x, max(0, x-50), -1):
            if sample_y < gray.shape[0] and scan_x < gray.shape[1]:
                if gray[sample_y, scan_x] < 40:
                    thickness += 1
                else:
                    break
        if thickness > 0:
            thickness_samples.append(thickness)
    
    # Return average or default
    return int(np.mean(thickness_samples)) if thickness_samples else 20

def calculate_bounds_from_edges(gray, edges, scan_depth, threshold):
    """Calculate inner bounds from edge detection"""
    h, w = gray.shape
    x1, y1, x2, y2 = 0, 0, w, h
    
    # Scan deeper to find actual bounds
    if 'top' in edges:
        for y in range(min(h//3, 200)):
            if np.mean(gray[y, :] < threshold) < 0.5:
                y1 = y
                break
    
    if 'bottom' in edges:
        for y in range(h-1, max(h*2//3, h-200), -1):
            if np.mean(gray[y, :] < threshold) < 0.5:
                y2 = y
                break
    
    if 'left' in edges:
        for x in range(min(w//3, 200)):
            if np.mean(gray[:, x] < threshold) < 0.5:
                x1 = x
                break
    
    if 'right' in edges:
        for x in range(w-1, max(w*2//3, w-200), -1):
            if np.mean(gray[:, x] < threshold) < 0.5:
                x2 = x
                break
    
    # Validate bounds
    if x2 > x1 + 100 and y2 > y1 + 100:
        return (x1, y1, x2, y2)
    
    return None

def validate_rectangular_mask(mask, img_w, img_h):
    """Validate if detected mask is rectangular and reasonable"""
    if not mask or 'bounds' not in mask:
        return False
    
    x1, y1, x2, y2 = mask['bounds']
    w = x2 - x1
    h = y2 - y1
    
    # Size constraints
    if w < 100 or h < 100:
        return False
    
    # Aspect ratio constraints
    aspect_ratio = w / h
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False
    
    # Must be smaller than image
    if w >= img_w * 0.95 or h >= img_h * 0.95:
        return False
    
    return True

def calculate_mask_score(mask, gray):
    """Calculate quality score for detected mask"""
    x1, y1, x2, y2 = mask['bounds']
    
    # Score based on multiple factors
    score = 0
    
    # 1. Size score (prefer reasonable sizes)
    area = (x2 - x1) * (y2 - y1)
    total_area = gray.shape[0] * gray.shape[1]
    area_ratio = area / total_area
    
    if 0.1 < area_ratio < 0.7:
        score += 30
    
    # 2. Type score (prefer center rectangles for center masking)
    if mask['type'] == 'center_rectangle':
        score += 20
    elif mask['type'] == 'contour_based':
        score += 15
    
    # 3. Edge contrast score
    # Check if there's clear contrast at mask boundaries
    margin = 10
    if x1 >= margin and y1 >= margin and x2 < gray.shape[1] - margin and y2 < gray.shape[0] - margin:
        # Inner region (should be brighter)
        inner_region = gray[y1+margin:y2-margin, x1+margin:x2-margin]
        # Outer region (should be darker at edges)
        outer_regions = [
            gray[max(0, y1-margin):y1, x1:x2],  # top
            gray[y2:min(gray.shape[0], y2+margin), x1:x2],  # bottom
            gray[y1:y2, max(0, x1-margin):x1],  # left
            gray[y1:y2, x2:min(gray.shape[1], x2+margin)]  # right
        ]
        
        inner_mean = np.mean(inner_region)
        outer_means = [np.mean(region) for region in outer_regions if region.size > 0]
        
        if outer_means and inner_mean > np.mean(outer_means) + 50:
            score += 30
    
    return score

def apply_replicate_inpainting(original, mask_bounds):
    """Apply Replicate AI inpainting to remove masking"""
    if not REPLICATE_API_TOKEN:
        log_debug("No Replicate API token, using fallback")
        return None
    
    try:
        import replicate
        
        x1, y1, x2, y2 = mask_bounds
        h, w = original.shape[:2]
        
        # Create mask for inpainting
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add buffer around detected bounds for better inpainting
        buffer = 30
        mask_x1 = max(0, x1 - buffer)
        mask_y1 = max(0, y1 - buffer)
        mask_x2 = min(w, x2 + buffer)
        mask_y2 = min(h, y2 + buffer)
        
        # Mark the masking area (including the black lines)
        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
        
        # But protect the inner content
        inner_buffer = 5
        mask[y1+inner_buffer:y2-inner_buffer, x1+inner_buffer:x2-inner_buffer] = 0
        
        # Convert images for API
        image_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Save temporary files
        image_io = io.BytesIO()
        mask_io = io.BytesIO()
        image_pil.save(image_io, format='PNG')
        mask_pil.save(mask_io, format='PNG')
        image_io.seek(0)
        mask_io.seek(0)
        
        # Run inpainting
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting",
            input={
                "image": image_io,
                "mask": mask_io,
                "prompt": "clean white seamless background, product photography background",
                "negative_prompt": "black, dark, shadows, borders, frames, lines",
                "num_inference_steps": 25,
                "guidance_scale": 7.5
            }
        )
        
        if output and len(output) > 0:
            # Download result
            response = requests.get(output[0])
            result_image = Image.open(io.BytesIO(response.content))
            result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            
            # Ensure wedding ring area is preserved
            result[y1:y2, x1:x2] = original[y1:y2, x1:x2]
            
            return result
            
    except Exception as e:
        log_debug(f"Replicate inpainting failed: {str(e)}")
    
    return None

def apply_simple_masking_removal(image, mask_bounds):
    """Simple masking removal using averaging and blending"""
    x1, y1, x2, y2 = mask_bounds
    h, w = image.shape[:2]
    result = image.copy()
    
    # Get background color from edges
    edge_pixels = []
    margin = 100
    
    # Sample from all edges
    if y1 > margin:
        edge_pixels.extend(image[0:margin, :].reshape(-1, 3))
    if y2 < h - margin:
        edge_pixels.extend(image[h-margin:h, :].reshape(-1, 3))
    if x1 > margin:
        edge_pixels.extend(image[:, 0:margin].reshape(-1, 3))
    if x2 < w - margin:
        edge_pixels.extend(image[:, w-margin:w].reshape(-1, 3))
    
    if edge_pixels:
        bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
    else:
        bg_color = np.array([240, 240, 240], dtype=np.uint8)
    
    # Fill the masking area
    buffer = 25
    
    # Top masking
    if y1 > buffer:
        result[max(0, y1-buffer):y1, x1:x2] = bg_color
    
    # Bottom masking
    if y2 < h - buffer:
        result[y2:min(h, y2+buffer), x1:x2] = bg_color
    
    # Left masking
    if x1 > buffer:
        result[y1:y2, max(0, x1-buffer):x1] = bg_color
    
    # Right masking
    if x2 < w - buffer:
        result[y1:y2, x2:min(w, x2+buffer)] = bg_color
    
    # Smooth blending
    for i in range(3):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    
    # Ensure ring is preserved
    center_result = result.copy()
    center_result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    # Blend edges
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    
    for c in range(3):
        result[:,:,c] = (center_result[:,:,c] * mask + result[:,:,c] * (1 - mask)).astype(np.uint8)
    
    return result

def enhance_wedding_ring_details(image, metal_type, is_ring_region=True):
    """Enhanced detail processing for wedding rings"""
    result = image.copy()
    
    # Different enhancement levels for ring vs background
    if is_ring_region:
        # Strong enhancement for ring details
        brightness_factor = 1.25
        contrast_factor = 1.20
        sharpness_factor = 1.45
        
        # Apply CLAHE for better detail
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
    else:
        # Light enhancement for background
        brightness_factor = 1.12
        contrast_factor = 1.08
        sharpness_factor = 1.0
    
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    # Brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    # Sharpness (only for ring)
    if is_ring_region and sharpness_factor > 1.0:
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(sharpness_factor)
    
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Metal-specific adjustments for ring region only
    if is_ring_region:
        if metal_type in ["white_gold", "champagne_gold"]:
            # Add white overlay for whiter appearance
            white_overlay = np.ones_like(result) * 255
            result = cv2.addWeighted(result, 0.95, white_overlay, 0.05, 0)
    
    return result

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create thumbnail with ring filling the frame"""
    h, w = image.shape[:2]
    
    # Convert to grayscale for ring detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the ring using multiple methods
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (likely the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_ring, h_ring = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = int(max(w_ring, h_ring) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_ring = min(w - x, w_ring + 2 * padding)
        h_ring = min(h - y, h_ring + 2 * padding)
        
        # Crop to ring area
        cropped = image[y:y+h_ring, x:x+w_ring]
    else:
        # Fallback: use center crop
        crop_size = int(min(w, h) * 0.8)
        x = (w - crop_size) // 2
        y = (h - crop_size) // 2
        cropped = image[y:y+crop_size, x:x+crop_size]
    
    # Resize to target
    thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """Main handler function for RunPod"""
    try:
        log_debug("Handler started")
        
        # Parse input - support multiple formats
        job_input = event.get("input", {})
        
        # Try different possible input formats
        image_base64 = None
        if "image" in job_input:
            image_base64 = job_input["image"]
        elif "image_base64" in job_input:
            image_base64 = job_input["image_base64"]
        elif "enhanced_image" in job_input:
            image_base64 = job_input["enhanced_image"]
        
        if not image_base64:
            log_debug(f"Available keys: {list(job_input.keys())}")
            return {"output": {"error": "No image provided", "available_keys": list(job_input.keys())}}
        
        # Decode image
        image = decode_base64_image(image_base64)
        log_debug(f"Image decoded: {image.shape}")
        
        # Step 1: Detect masking
        mask_info = detect_masking_comprehensive(image)
        
        if mask_info:
            log_debug(f"Masking detected: {mask_info['type']}")
            mask_bounds = mask_info['bounds']
            
            # Step 2: Detect metal type within masked area
            metal_type = detect_metal_type_in_region(image, mask_bounds)
            log_debug(f"Metal type detected: {metal_type}")
            
            # Step 3: Enhance ring details within mask
            x1, y1, x2, y2 = mask_bounds
            ring_region = image[y1:y2, x1:x2].copy()
            enhanced_ring = enhance_wedding_ring_details(ring_region, metal_type, is_ring_region=True)
            
            # Step 4: Remove masking
            if REPLICATE_API_TOKEN:
                result = apply_replicate_inpainting(image, mask_bounds)
                if result is None:
                    result = apply_simple_masking_removal(image, mask_bounds)
            else:
                result = apply_simple_masking_removal(image, mask_bounds)
            
            # Put enhanced ring back
            result[y1:y2, x1:x2] = enhanced_ring
            
            # Step 5: Light enhancement for full image
            result = enhance_wedding_ring_details(result, metal_type, is_ring_region=False)
            
        else:
            log_debug("No masking detected, applying direct enhancement")
            # No masking, just enhance
            metal_type = detect_metal_type_in_region(image)
            result = enhance_wedding_ring_details(image, metal_type, is_ring_region=True)
        
        # Create thumbnail
        thumbnail = create_thumbnail(result)
        
        # Encode results
        enhanced_base64 = encode_image_to_base64(result)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        # Return with correct structure for Make.com
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "masking_detected": mask_info is not None,
                    "masking_type": mask_info['type'] if mask_info else None,
                    "version": "v118"
                }
            }
        }
        
    except Exception as e:
        log_debug(f"Error in handler: {str(e)}")
        import traceback
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
