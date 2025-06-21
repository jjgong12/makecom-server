#!/usr/bin/env python3
import runpod
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import io
import base64
import sys
import traceback
import replicate
import os
from collections import Counter

# Initialize Replicate
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
if replicate_api_token:
    replicate.Client(api_token=replicate_api_token)

def log_debug(message):
    """Enhanced debug logging"""
    print(f"[DEBUG v120] {message}", file=sys.stderr)

def calculate_bounds_from_edges(gray, edges_with_black, scan_depth, threshold):
    """Calculate exact bounds of black borders"""
    h, w = gray.shape
    
    # Initialize with full image bounds
    left, top, right, bottom = 0, 0, w, h
    
    # Find exact boundaries for each edge
    if 'top' in edges_with_black:
        for y in range(min(scan_depth * 2, h)):
            if np.mean(gray[y, :] < threshold) < 0.5:
                top = y
                break
    
    if 'bottom' in edges_with_black:
        for y in range(h-1, max(h-scan_depth*2, -1), -1):
            if np.mean(gray[y, :] < threshold) < 0.5:
                bottom = y + 1
                break
    
    if 'left' in edges_with_black:
        for x in range(min(scan_depth * 2, w)):
            if np.mean(gray[:, x] < threshold) < 0.5:
                left = x
                break
    
    if 'right' in edges_with_black:
        for x in range(w-1, max(w-scan_depth*2, -1), -1):
            if np.mean(gray[:, x] < threshold) < 0.5:
                right = x + 1
                break
    
    # Validate bounds
    if left < right and top < bottom:
        return (left, top, right, bottom)
    return None

def estimate_line_thickness(gray, bounds):
    """Estimate the thickness of black lines"""
    left, top, right, bottom = bounds
    h, w = gray.shape
    thicknesses = []
    
    # Sample from multiple points
    # Top edge
    if top > 0:
        for x in range(left + 10, right - 10, 50):
            for y in range(top, min(top + 100, h)):
                if gray[y, x] > 50:
                    thicknesses.append(y - top)
                    break
    
    # Left edge
    if left > 0:
        for y in range(top + 10, bottom - 10, 50):
            for x in range(left, min(left + 100, w)):
                if gray[y, x] > 50:
                    thicknesses.append(x - left)
                    break
    
    if thicknesses:
        # Return median thickness
        return int(np.median(thicknesses))
    return 20  # Default

def validate_rectangular_mask(mask, img_w, img_h):
    """Validate if the mask represents a proper rectangular masking"""
    bounds = mask['bounds']
    x1, y1, x2, y2 = bounds
    
    # Check minimum size (at least 5% of image)
    mask_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    
    if mask_area < img_area * 0.05:
        return False
    
    # Check if it's too large (more than 90% would mean almost entire image)
    if mask_area > img_area * 0.9:
        return False
    
    # Check aspect ratio (should be somewhat rectangular, not too thin)
    aspect_ratio = (x2 - x1) / (y2 - y1) if y2 > y1 else 1
    if aspect_ratio < 0.2 or aspect_ratio > 5:
        return False
    
    return True

def calculate_mask_score(mask, gray):
    """Calculate a score for how likely this is the actual masking"""
    bounds = mask['bounds']
    x1, y1, x2, y2 = bounds
    h, w = gray.shape
    
    score = 0
    
    # Size score (prefer reasonable sizes)
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    if 0.1 < area_ratio < 0.8:
        score += 50
    
    # Type score
    if mask['type'] == 'center_rectangle':
        score += 30  # Prefer center rectangles as user reported
    elif mask['type'] == 'cross_validated':
        score += 40  # Prefer cross-validated detections
    
    # Edge contrast score
    if x1 > 0 and y1 > 0 and x2 < w and y2 < h:
        # Check if there's strong contrast at boundaries
        edge_mean = np.mean([
            np.mean(gray[y1-5:y1+5, x1:x2]),
            np.mean(gray[y1:y2, x1-5:x1+5])
        ])
        inner_mean = np.mean(gray[y1+10:y2-10, x1+10:x2-10])
        contrast = abs(inner_mean - edge_mean)
        score += min(contrast / 2, 30)
    
    return score

def detect_metal_type_conservative(region):
    """Conservative metal type detection - default to white when uncertain"""
    if region.size == 0:
        return "white"
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    avg_saturation = np.mean(s)
    avg_brightness = np.mean(v)
    
    # RGB analysis
    b, g, r = cv2.split(region)
    avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)
    
    # Metal type detection with conservative approach
    if avg_saturation < 15 and avg_brightness > 180:
        return "white"
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
            return "white"  # Conservative default

def detect_masking_cross_validation(image):
    """Enhanced cross-validation masking detection"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    log_debug(f"Starting cross-validation detection for {w}x{h} image")
    
    # Store all detected masks from different methods
    all_detections = []
    
    # Method 1: Progressive Edge Scanning (multiple thresholds)
    for threshold in [20, 30, 40, 50, 60, 70, 80]:
        masks = detect_edge_based_progressive(gray, w, h, threshold)
        all_detections.extend(masks)
    
    # Method 2: Center-based Detection (for masks not touching edges)
    center_masks = detect_center_based_enhanced(gray, w, h)
    all_detections.extend(center_masks)
    
    # Method 3: Gradient-based Detection
    gradient_masks = detect_gradient_based(gray, w, h)
    all_detections.extend(gradient_masks)
    
    # Method 4: Morphological Detection
    morph_masks = detect_morphological_based(gray, w, h)
    all_detections.extend(morph_masks)
    
    # Method 5: Color-based Detection (for very dark masks)
    color_masks = detect_color_based(image)
    all_detections.extend(color_masks)
    
    log_debug(f"Total detections from all methods: {len(all_detections)}")
    
    # Cross-validate: Find masks detected by multiple methods
    validated_masks = []
    
    # Group similar detections
    for i, mask1 in enumerate(all_detections):
        if not validate_rectangular_mask(mask1, w, h):
            continue
            
        x1_1, y1_1, x2_1, y2_1 = mask1['bounds']
        similar_count = 1
        combined_methods = [mask1['method']]
        
        for j, mask2 in enumerate(all_detections):
            if i == j:
                continue
                
            x1_2, y1_2, x2_2, y2_2 = mask2['bounds']
            
            # Check if masks are similar (IoU > 0.7)
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union_area = area1 + area2 - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > 0.7:
                    similar_count += 1
                    if mask2.get('method') not in combined_methods:
                        combined_methods.append(mask2['method'])
        
        if similar_count >= 2:  # Detected by at least 2 methods
            validated_masks.append({
                'bounds': mask1['bounds'],
                'type': 'cross_validated',
                'confidence': similar_count,
                'methods': combined_methods
            })
    
    # Remove duplicates from validated masks
    unique_masks = []
    for mask in validated_masks:
        is_duplicate = False
        for existing in unique_masks:
            if mask['bounds'] == existing['bounds']:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_masks.append(mask)
    
    log_debug(f"Cross-validated masks: {len(unique_masks)}")
    
    # If no cross-validated masks, fall back to best single detection
    if not unique_masks:
        log_debug("No cross-validated masks found, using best single detection")
        best_score = 0
        best_mask = None
        
        for mask in all_detections:
            if validate_rectangular_mask(mask, w, h):
                score = calculate_mask_score(mask, gray)
                if score > best_score:
                    best_score = score
                    best_mask = mask
        
        if best_mask:
            return best_mask
        return None
    
    # Choose best cross-validated mask
    best_score = 0
    best_mask = None
    
    for mask in unique_masks:
        score = calculate_mask_score(mask, gray) + (mask['confidence'] * 20)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    if best_mask:
        log_debug(f"Best mask: {best_mask['type']} with confidence {best_mask.get('confidence', 0)}")
    
    return best_mask

def detect_edge_based_progressive(gray, w, h, threshold):
    """Progressive edge-based detection with specific threshold"""
    masks = []
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for scan_pct in scan_percentages:
        scan_depth = int(min(w, h) * scan_pct)
        
        # Check each edge
        edges_with_black = []
        
        # Top edge
        if np.mean(gray[:scan_depth, :] < threshold) > 0.7:
            edges_with_black.append('top')
        
        # Bottom edge
        if np.mean(gray[-scan_depth:, :] < threshold) > 0.7:
            edges_with_black.append('bottom')
        
        # Left edge
        if np.mean(gray[:, :scan_depth] < threshold) > 0.7:
            edges_with_black.append('left')
        
        # Right edge
        if np.mean(gray[:, -scan_depth:] < threshold) > 0.7:
            edges_with_black.append('right')
        
        if len(edges_with_black) >= 2:
            bounds = calculate_bounds_from_edges(gray, edges_with_black, scan_depth, threshold)
            if bounds:
                masks.append({
                    'bounds': bounds,
                    'type': 'edge_based',
                    'method': f'edge_progressive_t{threshold}',
                    'edges': edges_with_black
                })
                break  # Found with this scan percentage
    
    return masks

def detect_center_based_enhanced(gray, w, h):
    """Enhanced center-based detection"""
    masks = []
    
    # Try different center regions
    center_regions = [
        (0.1, 0.9),  # 80% center
        (0.05, 0.95),  # 90% center
        (0.15, 0.85),  # 70% center
    ]
    
    for start_pct, end_pct in center_regions:
        center_x1 = int(w * start_pct)
        center_x2 = int(w * end_pct)
        center_y1 = int(h * start_pct)
        center_y2 = int(h * end_pct)
        
        center_region = gray[center_y1:center_y2, center_x1:center_x2]
        
        # Multiple thresholds
        for threshold in [30, 40, 50, 60]:
            _, black_mask = cv2.threshold(center_region, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up with morphology
            kernel = np.ones((5,5), np.uint8)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            
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
                        
                        # Estimate and account for line thickness
                        thickness = estimate_line_thickness(gray, (x, y, x + w_box, y + h_box))
                        
                        inner_x = x + thickness
                        inner_y = y + thickness
                        inner_w = w_box - 2 * thickness
                        inner_h = h_box - 2 * thickness
                        
                        if inner_w > 50 and inner_h > 50:  # Minimum size
                            masks.append({
                                'bounds': (inner_x, inner_y, inner_x + inner_w, inner_y + inner_h),
                                'type': 'center_rectangle',
                                'method': f'center_enhanced_t{threshold}',
                                'thickness': thickness
                            })
    
    return masks

def detect_gradient_based(gray, w, h):
    """Gradient-based detection for sharp edges"""
    masks = []
    
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    
    # Threshold gradient magnitude
    _, grad_binary = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(grad_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (w * h * 0.05):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Check if it's actually a dark region inside
                roi = gray[y:y+h_box, x:x+w_box]
                if np.mean(roi) < 60:  # Dark inside
                    thickness = 20  # Default
                    inner_bounds = (x + thickness, y + thickness, 
                                  x + w_box - thickness, y + h_box - thickness)
                    
                    masks.append({
                        'bounds': inner_bounds,
                        'type': 'gradient_based',
                        'method': 'gradient_sobel'
                    })
    
    return masks

def detect_morphological_based(gray, w, h):
    """Morphological operations to find rectangular structures"""
    masks = []
    
    # Try different thresholds
    for threshold in [40, 50, 60]:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel_sizes = [(10, 10), (15, 15), (20, 20)]
        
        for kernel_size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (w * h * 0.05):
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # Verify it's actually a masking
                    roi = gray[y:y+h_box, x:x+w_box]
                    if np.mean(roi) < 70:
                        thickness = 20
                        inner_bounds = (x + thickness, y + thickness,
                                      x + w_box - thickness, y + h_box - thickness)
                        
                        masks.append({
                            'bounds': inner_bounds,
                            'type': 'morphological',
                            'method': f'morph_t{threshold}_k{kernel_size[0]}'
                        })
    
    return masks

def detect_color_based(image):
    """Color-based detection for very dark masks"""
    masks = []
    h, w = image.shape[:2]
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # HSV: Low value (darkness)
    v_channel = hsv[:, :, 2]
    _, dark_mask_v = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY_INV)
    
    # LAB: Low lightness
    l_channel = lab[:, :, 0]
    _, dark_mask_l = cv2.threshold(l_channel, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks
    dark_mask = cv2.bitwise_and(dark_mask_v, dark_mask_l)
    
    # Clean up
    kernel = np.ones((10, 10), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (w * h * 0.05):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                thickness = 20
                inner_bounds = (x + thickness, y + thickness,
                              x + w_box - thickness, y + h_box - thickness)
                
                masks.append({
                    'bounds': inner_bounds,
                    'type': 'color_based',
                    'method': 'hsv_lab_combined'
                })
    
    return masks

def remove_masking_with_replicate(image, masking_bounds):
    """Remove masking using Replicate API with stable diffusion inpainting"""
    if not replicate_api_token:
        log_debug("No Replicate API token, returning original image")
        return image
    
    try:
        x1, y1, x2, y2 = masking_bounds
        h, w = image.shape[:2]
        
        # Create mask for inpainting (white = areas to inpaint)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mark the masking area to be removed
        # Expand bounds slightly to ensure complete removal
        expansion = 10
        mask_x1 = max(0, x1 - expansion)
        mask_y1 = max(0, y1 - expansion)
        mask_x2 = min(w, x2 + expansion)
        mask_y2 = min(h, y2 + expansion)
        
        # Draw the rectangular frame to be removed
        thickness = 30  # Thickness of the mask
        cv2.rectangle(mask, (mask_x1, mask_y1), (mask_x2, mask_y2), 255, thickness)
        
        # Convert images to base64
        # Original image
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Mask
        mask_pil = Image.fromarray(mask)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Run inpainting
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting",
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "prompt": "clean white seamless background, professional product photography background",
                "negative_prompt": "black, dark, shadows, borders, frames, lines, edges",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        
        if output and len(output) > 0:
            # Get the inpainted image
            result_url = output[0]
            import requests
            response = requests.get(result_url)
            
            if response.status_code == 200:
                result_image = Image.open(io.BytesIO(response.content))
                result_array = np.array(result_image)
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
                
                log_debug("Successfully removed masking with Replicate")
                return result_bgr
            else:
                log_debug(f"Failed to download inpainted image: {response.status_code}")
        else:
            log_debug("No output from Replicate API")
            
    except Exception as e:
        log_debug(f"Error in Replicate inpainting: {str(e)}")
        traceback.print_exc()
    
    # Fallback to original image
    return image

def apply_enhancements(image, metal_type):
    """Apply enhancements based on metal type"""
    # Parameter sets for different metal types
    params = {
        "rose_gold": {
            "brightness": 1.20,
            "contrast": 1.15,
            "saturation": 1.15,
            "sharpness": 1.40,
            "clarity_strength": 0.6,
            "white_overlay": 0.08,
            "color_temp": 0,
            "vibrance": 1.1
        },
        "yellow_gold": {
            "brightness": 1.22,
            "contrast": 1.18,
            "saturation": 1.10,
            "sharpness": 1.45,
            "clarity_strength": 0.7,
            "white_overlay": 0.10,
            "color_temp": -2,
            "vibrance": 1.08
        },
        "white_gold": {
            "brightness": 1.25,
            "contrast": 1.20,
            "saturation": 0.95,
            "sharpness": 1.50,
            "clarity_strength": 0.8,
            "white_overlay": 0.12,
            "color_temp": 2,
            "vibrance": 1.0
        },
        "white": {
            "brightness": 1.28,
            "contrast": 1.22,
            "saturation": 0.90,
            "sharpness": 1.55,
            "clarity_strength": 0.85,
            "white_overlay": 0.15,
            "color_temp": 3,
            "vibrance": 0.95
        }
    }
    
    # Get parameters for metal type
    p = params.get(metal_type, params["white"])
    
    # Convert to PIL for enhancement
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply brightness
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(p["brightness"])
    
    # Apply contrast
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(p["contrast"])
    
    # Apply sharpness
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(p["sharpness"])
    
    # Convert back to numpy
    enhanced = np.array(img_pil)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE for local contrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Add white overlay
    if p["white_overlay"] > 0:
        white_layer = np.full_like(enhanced, 255, dtype=np.uint8)
        enhanced = cv2.addWeighted(enhanced, 1 - p["white_overlay"], 
                                 white_layer, p["white_overlay"], 0)
    
    return enhanced

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create a properly sized thumbnail"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling to fit the ring optimally
    scale = min(target_w / w, target_h / h) * 0.95  # 95% to leave small margin
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white canvas
    canvas = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def handler(event):
    """Main handler function for RunPod - FIXED FOR MAKE.COM"""
    try:
        log_debug("=== Starting v120 Handler Processing ===")
        
        # Get input - Make.com sends data in event["input"]
        job_input = event.get("input", {})
        log_debug(f"Available keys in input: {list(job_input.keys())}")
        
        # Try to get image from multiple possible fields
        image_base64 = None
        
        # Method 1: Direct 'image' field
        if "image" in job_input:
            image_base64 = job_input["image"]
            log_debug("Found image in 'image' field")
        
        # Method 2: 'image_base64' field
        elif "image_base64" in job_input:
            image_base64 = job_input["image_base64"]
            log_debug("Found image in 'image_base64' field")
        
        # Method 3: Check if there's nested structure
        elif "data" in job_input and isinstance(job_input["data"], dict):
            if "image" in job_input["data"]:
                image_base64 = job_input["data"]["image"]
                log_debug("Found image in 'data.image' field")
            elif "image_base64" in job_input["data"]:
                image_base64 = job_input["data"]["image_base64"]
                log_debug("Found image in 'data.image_base64' field")
        
        if not image_base64:
            log_debug(f"No image found. Full input structure: {job_input}")
            return {
                "output": {
                    "error": "No image provided in 'image' or 'image_base64' field",
                    "status": "error",
                    "available_keys": list(job_input.keys())
                }
            }
        
        # Decode image - DO NOT add padding
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        original_shape = img_array.shape
        log_debug(f"Original image shape: {original_shape}")
        
        # Detect masking with cross-validation
        masking_info = detect_masking_cross_validation(img_array)
        
        if masking_info:
            log_debug(f"Masking detected: {masking_info}")
            
            # Extract the ring region for metal detection
            x1, y1, x2, y2 = masking_info['bounds']
            ring_region = img_array[y1:y2, x1:x2]
            
            # Detect metal type from the ring region
            metal_type = detect_metal_type_conservative(ring_region)
            log_debug(f"Detected metal type: {metal_type}")
            
            # Remove the masking
            img_array = remove_masking_with_replicate(img_array, (x1, y1, x2, y2))
            
            # Apply enhancements to the whole image
            enhanced = apply_enhancements(img_array, metal_type)
        else:
            log_debug("No masking detected")
            
            # Detect metal type from whole image
            metal_type = detect_metal_type_conservative(img_array)
            log_debug(f"Detected metal type: {metal_type}")
            
            # Apply enhancements
            enhanced = apply_enhancements(img_array, metal_type)
        
        # Create thumbnail
        thumbnail = create_thumbnail(enhanced)
        
        # Convert to base64
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        enhanced_buffer = io.BytesIO()
        enhanced_pil.save(enhanced_buffer, format='PNG', quality=95)
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode()
        
        thumbnail_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        thumbnail_buffer = io.BytesIO()
        thumbnail_pil.save(thumbnail_buffer, format='PNG', quality=95)
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode()
        
        # Return with proper nesting for Make.com
        # Make.com expects: {{4.data.output.output.enhanced_image}}
        return {
            "output": {
                "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
                "thumbnail": f"data:image/png;base64,{thumbnail_base64}",
                "metal_type": metal_type,
                "masking_detected": masking_info is not None,
                "masking_info": str(masking_info) if masking_info else "None",
                "original_size": f"{original_shape[1]}x{original_shape[0]}",
                "status": "success"
            }
        }
        
    except Exception as e:
        log_debug(f"Error in processing: {str(e)}")
        traceback.print_exc()
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler - CORRECT FORMAT
runpod.serverless.start(handler)
