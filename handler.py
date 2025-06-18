import runpod
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter

def handler(event):
    try:
        # Input validation
        if "input" not in event:
            return {"error": "No input provided"}
        
        image_base64 = event["input"].get("image_base64")
        if not image_base64:
            return {"error": "No image_base64 provided"}
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image"}
        
        # Step 1: ULTRA AGGRESSIVE BLACK BORDER DETECTION AND REMOVAL
        # 가장자리별로 검은 테두리 두께 측정
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 각 방향에서 검은 테두리 두께 측정 (최대 200픽셀까지)
        def find_border_thickness(img_gray, direction='top'):
            max_check = 200  # 최대 200픽셀까지 확인
            
            if direction == 'top':
                for i in range(min(max_check, h)):
                    row = img_gray[i, :]
                    if np.mean(row) > 50:  # 검은색이 아닌 부분 발견
                        return i
            elif direction == 'bottom':
                for i in range(min(max_check, h)):
                    row = img_gray[h-1-i, :]
                    if np.mean(row) > 50:
                        return i
            elif direction == 'left':
                for i in range(min(max_check, w)):
                    col = img_gray[:, i]
                    if np.mean(col) > 50:
                        return i
            elif direction == 'right':
                for i in range(min(max_check, w)):
                    col = img_gray[:, w-1-i]
                    if np.mean(col) > 50:
                        return i
            return 0
        
        # 각 방향에서 테두리 두께 측정
        top_border = find_border_thickness(gray, 'top')
        bottom_border = find_border_thickness(gray, 'bottom')
        left_border = find_border_thickness(gray, 'left')
        right_border = find_border_thickness(gray, 'right')
        
        # 안전 마진 추가 (측정값 + 10픽셀)
        margin = 10
        top_border += margin
        bottom_border += margin
        left_border += margin
        right_border += margin
        
        # 크롭 영역 계산
        x1 = min(left_border, w-1)
        y1 = min(top_border, h-1)
        x2 = max(w - right_border, x1+1)
        y2 = max(h - bottom_border, y1+1)
        
        # 크롭 실행
        cropped = image[y1:y2, x1:x2]
        
        # 크롭된 이미지가 너무 작으면 원본 사용
        if cropped.shape[0] < 100 or cropped.shape[1] < 100:
            cropped = image
        
        # Step 2: 추가 검은색 제거 (혹시 남은 부분)
        # 다시 한번 검사
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # 여러 threshold로 검은색 영역 찾기
        masks = []
        for thresh in [30, 40, 50, 60, 70]:
            _, mask = cv2.threshold(gray_cropped, thresh, 255, cv2.THRESH_BINARY)
            masks.append(mask)
        
        # 모든 마스크 결합
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_and(combined_mask, mask)
        
        # 가장 큰 흰색 영역 찾기 (실제 콘텐츠)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 두 번째 크롭
            final_cropped = cropped[y:y+h, x:x+w]
        else:
            final_cropped = cropped
        
        # Step 3: 이미지 크기 확대 (너무 작으면)
        h, w = final_cropped.shape[:2]
        min_size = 2048
        
        if w < min_size or h < min_size:
            scale = min_size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            final_cropped = cv2.resize(final_cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 4: 반지 영역 감지
        # HSV로 변환하여 금속성 영역 찾기
        hsv = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2HSV)
        
        # 여러 범위로 금속 영역 감지
        metal_masks = []
        
        # 밝은 금속 (화이트골드, 실버)
        metal_masks.append(cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255])))
        
        # 골드 톤
        metal_masks.append(cv2.inRange(hsv, np.array([15, 30, 100]), np.array([35, 255, 255])))
        
        # 로즈골드 톤
        metal_masks.append(cv2.inRange(hsv, np.array([0, 20, 100]), np.array([20, 100, 255])))
        
        # 모든 마스크 결합
        ring_mask = metal_masks[0]
        for mask in metal_masks[1:]:
            ring_mask = cv2.bitwise_or(ring_mask, mask)
        
        # 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, kernel)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 영역만 선택
        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 면적이 큰 상위 2개 컨투어 선택 (반지 2개)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            ring_mask_clean = np.zeros_like(ring_mask)
            cv2.drawContours(ring_mask_clean, contours, -1, 255, -1)
            
            # 전체 바운딩 박스 계산
            x_min, y_min = final_cropped.shape[1], final_cropped.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            ring_bbox = {
                'x': x_min, 'y': y_min,
                'w': x_max - x_min, 'h': y_max - y_min,
                'cx': (x_min + x_max) // 2,
                'cy': (y_min + y_max) // 2
            }
        else:
            ring_mask_clean = np.ones_like(ring_mask) * 255
            h, w = final_cropped.shape[:2]
            ring_bbox = {'x': 0, 'y': 0, 'w': w, 'h': h, 'cx': w//2, 'cy': h//2}
        
        # Step 5: 금속 타입과 조명 감지
        metal_type, lighting = detect_metal_and_lighting(final_cropped, ring_mask_clean)
        
        # Step 6: v13.3 보정 적용
        params = get_v13_3_params(metal_type, lighting)
        
        # PIL로 변환
        rgb_image = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 보정 적용
        enhanced = pil_image.copy()
        
        # Brightness
        if params['brightness'] != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(params['brightness'])
        
        # Contrast
        if params['contrast'] != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
        
        # Sharpness
        if params['sharpness'] != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
        
        # Convert to numpy for advanced processing
        enhanced_np = np.array(enhanced)
        
        # Color temperature adjustment
        if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
            lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,1] = np.clip(lab[:,:,1] + params['color_temp_a'], 0, 255)
            lab[:,:,2] = np.clip(lab[:,:,2] + params['color_temp_b'], 0, 255)
            enhanced_np = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # White overlay on ring area
        if params['white_overlay'] > 0 and np.any(ring_mask_clean > 0):
            white_layer = np.ones_like(enhanced_np) * 255
            ring_mask_3ch = cv2.cvtColor(ring_mask_clean, cv2.COLOR_GRAY2RGB) / 255.0
            overlay_mask = ring_mask_3ch * params['white_overlay']
            enhanced_np = (enhanced_np * (1 - overlay_mask) + white_layer * overlay_mask).astype(np.uint8)
        
        # Blend with original
        original_np = np.array(pil_image)
        if params['original_blend'] > 0:
            enhanced_np = (enhanced_np * (1 - params['original_blend']) + original_np * params['original_blend']).astype(np.uint8)
        
        # Convert back to PIL
        final_pil = Image.fromarray(enhanced_np)
        
        # Apply final sharpening
        final_pil = final_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=0))
        
        # Step 7: 썸네일 생성 (반지가 80% 차지하도록)
        thumbnail = create_perfect_thumbnail(final_pil, ring_bbox)
        
        # Convert to base64
        # Main image
        main_buffer = io.BytesIO()
        final_pil.save(main_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # Thumbnail
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        # Return with output nesting
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "original_size": f"{image.shape[1]}x{image.shape[0]}",
                    "border_removed": {
                        "top": top_border,
                        "bottom": bottom_border,
                        "left": left_border,
                        "right": right_border
                    },
                    "final_size": f"{final_pil.width}x{final_pil.height}",
                    "thumbnail_size": "1000x1300",
                    "ring_area_ratio": "80%",
                    "black_border_removed": True,
                    "enhancement_applied": True,
                    "version": "v21.0",
                    "status": "success"
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": f"Processing error: {str(e)}",
                "status": "failed",
                "version": "v21.0"
            }
        }

def detect_metal_and_lighting(image_np, ring_mask):
    """Detect metal type and lighting condition"""
    try:
        # Get ring pixels only
        if np.any(ring_mask > 128):
            ring_pixels = image_np[ring_mask > 128]
            avg_color = np.mean(ring_pixels, axis=0)
        else:
            # Use center area
            h, w = image_np.shape[:2]
            center = image_np[h//3:2*h//3, w//3:2*w//3]
            avg_color = np.mean(center.reshape(-1, 3), axis=0)
        
        # BGR to RGB
        b, g, r = avg_color
        
        # Brightness
        brightness = np.mean([r, g, b])
        
        # Color characteristics
        rg_diff = abs(r - g)
        gb_diff = abs(g - b)
        rb_diff = abs(r - b)
        
        # Detect metal type
        if rg_diff < 15 and gb_diff < 15 and rb_diff < 15 and brightness > 180:
            metal_type = "white_gold"
        elif r > g > b and rg_diff < 30 and brightness > 150:
            metal_type = "rose_gold"
        elif r > g and r > b and rg_diff > 20:
            metal_type = "yellow_gold"
        else:
            metal_type = "mixed_metal"
        
        # Detect lighting
        if brightness > 200:
            lighting = "bright"
        elif brightness < 150:
            lighting = "ambient"
        else:
            lighting = "natural"
        
        return metal_type, lighting
        
    except:
        return "white_gold", "natural"

def get_v13_3_params(metal_type, lighting):
    """Get v13.3 enhancement parameters"""
    params_v13_3 = {
        'white_gold': {
            'bright': {
                'brightness': 1.22,
                'contrast': 1.18,
                'white_overlay': 0.11,
                'sharpness': 1.20,
                'color_temp_a': -5,
                'color_temp_b': -5,
                'original_blend': 0.10
            },
            'natural': {
                'brightness': 1.18,
                'contrast': 1.12,
                'white_overlay': 0.09,
                'sharpness': 1.15,
                'color_temp_a': -3,
                'color_temp_b': -3,
                'original_blend': 0.15
            },
            'ambient': {
                'brightness': 1.25,
                'contrast': 1.20,
                'white_overlay': 0.13,
                'sharpness': 1.18,
                'color_temp_a': -4,
                'color_temp_b': -4,
                'original_blend': 0.12
            }
        },
        'yellow_gold': {
            'bright': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.08,
                'sharpness': 1.18,
                'color_temp_a': 3,
                'color_temp_b': 2,
                'original_blend': 0.12
            },
            'natural': {
                'brightness': 1.18,
                'contrast': 1.10,
                'white_overlay': 0.06,
                'sharpness': 1.15,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.18
            },
            'ambient': {
                'brightness': 1.23,
                'contrast': 1.18,
                'white_overlay': 0.10,
                'sharpness': 1.17,
                'color_temp_a': 3,
                'color_temp_b': 2,
                'original_blend': 0.15
            }
        },
        'rose_gold': {
            'bright': {
                'brightness': 1.18,
                'contrast': 1.12,
                'white_overlay': 0.07,
                'sharpness': 1.17,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.15
            },
            'natural': {
                'brightness': 1.15,
                'contrast': 1.08,
                'white_overlay': 0.06,
                'sharpness': 1.15,
                'color_temp_a': 2,
                'color_temp_b': 1,
                'original_blend': 0.20
            },
            'ambient': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.09,
                'sharpness': 1.16,
                'color_temp_a': 3,
                'color_temp_b': 1,
                'original_blend': 0.18
            }
        },
        'mixed_metal': {
            'bright': {
                'brightness': 1.20,
                'contrast': 1.15,
                'white_overlay': 0.09,
                'sharpness': 1.18,
                'color_temp_a': -1,
                'color_temp_b': -1,
                'original_blend': 0.12
            },
            'natural': {
                'brightness': 1.17,
                'contrast': 1.10,
                'white_overlay': 0.07,
                'sharpness': 1.15,
                'color_temp_a': 0,
                'color_temp_b': 0,
                'original_blend': 0.17
            },
            'ambient': {
                'brightness': 1.22,
                'contrast': 1.17,
                'white_overlay': 0.11,
                'sharpness': 1.17,
                'color_temp_a': -1,
                'color_temp_b': -1,
                'original_blend': 0.15
            }
        }
    }
    
    if metal_type in params_v13_3 and lighting in params_v13_3[metal_type]:
        return params_v13_3[metal_type][lighting]
    else:
        return params_v13_3['white_gold']['natural']

def create_perfect_thumbnail(pil_image, ring_bbox):
    """Create 1000x1300 thumbnail with ring taking 80% of space"""
    target_width = 1000
    target_height = 1300
    
    # Extract ring area with small padding
    padding = 20  # 작은 패딩만
    x1 = max(0, ring_bbox['x'] - padding)
    y1 = max(0, ring_bbox['y'] - padding)
    x2 = min(pil_image.width, ring_bbox['x'] + ring_bbox['w'] + padding)
    y2 = min(pil_image.height, ring_bbox['y'] + ring_bbox['h'] + padding)
    
    # Crop to ring area
    ring_crop = pil_image.crop((x1, y1, x2, y2))
    
    # Calculate scale to make ring fill 80% of thumbnail
    crop_w, crop_h = ring_crop.size
    
    # Target size for ring (80% of thumbnail)
    target_ring_w = target_width * 0.8
    target_ring_h = target_height * 0.8
    
    # Calculate scale
    scale_w = target_ring_w / crop_w
    scale_h = target_ring_h / crop_h
    scale = min(scale_w, scale_h)
    
    # Resize ring
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized_ring = ring_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create white background
    thumbnail = Image.new('RGB', (target_width, target_height), 'white')
    
    # Center the ring
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    thumbnail.paste(resized_ring, (x_offset, y_offset))
    
    # Apply final sharpening
    thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=0))
    
    return thumbnail

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
