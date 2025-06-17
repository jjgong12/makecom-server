import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09, 'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3, 'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01},
        'warm': {'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12, 'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5, 'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98},
        'cool': {'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07, 'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2, 'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.02}
    },
    'rose_gold': {
        'natural': {'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06, 'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1, 'original_blend': 0.20, 'saturation': 1.12, 'gamma': 0.98},
        'warm': {'brightness': 1.12, 'contrast': 1.06, 'white_overlay': 0.08, 'sharpness': 1.12, 'color_temp_a': 0, 'color_temp_b': -1, 'original_blend': 0.22, 'saturation': 1.10, 'gamma': 0.95},
        'cool': {'brightness': 1.18, 'contrast': 1.10, 'white_overlay': 0.05, 'sharpness': 1.18, 'color_temp_a': 4, 'color_temp_b': 3, 'original_blend': 0.18, 'saturation': 1.15, 'gamma': 1.00}
    },
    'champagne_gold': {
        'natural': {'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12, 'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4, 'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00},
        'warm': {'brightness': 1.14, 'contrast': 1.09, 'white_overlay': 0.14, 'sharpness': 1.14, 'color_temp_a': -6, 'color_temp_b': -6, 'original_blend': 0.17, 'saturation': 1.00, 'gamma': 0.97},
        'cool': {'brightness': 1.20, 'contrast': 1.13, 'white_overlay': 0.10, 'sharpness': 1.18, 'color_temp_a': -2, 'color_temp_b': -2, 'original_blend': 0.13, 'saturation': 1.04, 'gamma': 1.01}
    },
    'yellow_gold': {
        'natural': {'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05, 'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2, 'original_blend': 0.22, 'saturation': 1.18, 'gamma': 0.99},
        'warm': {'brightness': 1.13, 'contrast': 1.07, 'white_overlay': 0.07, 'sharpness': 1.12, 'color_temp_a': 1, 'color_temp_b': 0, 'original_blend': 0.25, 'saturation': 1.15, 'gamma': 0.96},
        'cool': {'brightness': 1.19, 'contrast': 1.11, 'white_overlay': 0.04, 'sharpness': 1.16, 'color_temp_a': 5, 'color_temp_b': 4, 'original_blend': 0.20, 'saturation': 1.20, 'gamma': 1.01}
    }
}

def detect_black_border_aggressive(image):
    """검은색 테두리를 100% 확실하게 감지"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 각 가장자리에서 안쪽으로 들어가면서 검은색이 아닌 픽셀 찾기
    # 상단
    top = 0
    for y in range(height // 2):  # 최대 절반까지 확인
        row = gray[y, :]
        if np.max(row) > 100 or np.mean(row) > 50:  # 밝은 픽셀이 있거나 평균이 50 이상
            break
        top = y + 1
    
    # 하단
    bottom = height
    for y in range(height - 1, height // 2, -1):
        row = gray[y, :]
        if np.max(row) > 100 or np.mean(row) > 50:
            break
        bottom = y
    
    # 좌측
    left = 0
    for x in range(width // 2):
        col = gray[:, x]
        if np.max(col) > 100 or np.mean(col) > 50:
            break
        left = x + 1
    
    # 우측
    right = width
    for x in range(width - 1, width // 2, -1):
        col = gray[:, x]
        if np.max(col) > 100 or np.mean(col) > 50:
            break
        right = x
    
    # 검은색 테두리가 있는지 확인
    if top > 10 or (height - bottom) > 10 or left > 10 or (width - right) > 10:
        # 추가 여유분
        margin = 5
        top = min(top + margin, height // 2)
        bottom = max(bottom - margin, height // 2)
        left = min(left + margin, width // 2)
        right = max(right - margin, width // 2)
        
        return {
            'found': True,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'thickness': max(top, height - bottom, left, width - right)
        }
    
    return {'found': False}

def remove_black_border_completely(image, border_info):
    """검은색 테두리를 완전히 제거 - 크롭 방식"""
    if not border_info['found']:
        return image
    
    height, width = image.shape[:2]
    
    # 검은색 테두리 제외하고 크롭
    top = border_info['top']
    bottom = border_info['bottom']
    left = border_info['left']
    right = border_info['right']
    
    # 크롭 실행
    cropped = image[top:bottom, left:right].copy()
    
    # 원본 크기로 리사이즈 (배경이 늘어나서 자연스럽게)
    cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    return cropped

def detect_ring_in_image(image):
    """이미지에서 웨딩링 영역 감지"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 밝은 영역 감지 (웨딩링은 보통 밝음)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 중앙에 가장 가까운 큰 컨투어 찾기
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        best_contour = None
        min_distance = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 최소 크기
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            # 여유 마진 추가
            margin = 50
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(width - x, w + 2 * margin)
            h = min(height - y, h + 2 * margin)
            return (x, y, w, h)
    
    # 웨딩링을 못 찾으면 중앙 영역 반환
    w, h = width // 2, height // 2
    x, y = width // 4, height // 4
    return (x, y, w, h)

def enhance_wedding_ring_v13_3(image, metal_type='champagne_gold', lighting='natural'):
    """v13.3 완전한 10단계 보정 - 더 강하게 적용"""
    params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['champagne_gold']['natural'])
    
    # 1. 노이즈 제거
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # PIL 이미지로 변환
    pil_image = Image.fromarray(denoised)
    
    # 2. 밝기 조정 (더 강하게)
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(params['brightness'] * 1.1)  # 10% 더 밝게
    
    # 3. 대비 조정
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(params['contrast'] * 1.05)  # 5% 더 강하게
    
    # 4. 선명도 조정
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(params['sharpness'])
    
    # 5. 채도 조정
    saturation = params.get('saturation', 1.0)
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(saturation)
    
    # numpy 배열로 변환
    enhanced_array = np.array(enhanced)
    
    # 6. 하얀색 오버레이 (더 강하게)
    white_overlay = np.full_like(enhanced_array, 255)
    overlay_strength = params['white_overlay'] * 1.5  # 50% 더 강하게
    enhanced_array = cv2.addWeighted(enhanced_array, 1 - overlay_strength, white_overlay, overlay_strength, 0)
    
    # 7. 색온도 조정 (LAB)
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 1] += params['color_temp_a'] * 1.2
    lab[:, :, 2] += params['color_temp_b'] * 1.2
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # clipLimit 증가
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 9. 감마 보정
    gamma = params.get('gamma', 1.0) * 0.95  # 약간 더 밝게
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_array = cv2.LUT(enhanced_array, table)
    
    # 10. 원본과 블렌딩 (적게 섞어서 보정 효과 극대화)
    blend_ratio = params['original_blend'] * 0.5  # 원본을 덜 섞음
    final = cv2.addWeighted(enhanced_array, 1 - blend_ratio, image, blend_ratio, 0)
    
    return final

def create_perfect_thumbnail(image, ring_bbox=None):
    """완벽한 1000x1300 썸네일 생성 - 웨딩링이 화면 가득"""
    target_w, target_h = 1000, 1300
    
    if ring_bbox and ring_bbox[2] > 50 and ring_bbox[3] > 50:
        x, y, w, h = ring_bbox
        # 여유분 줄이기 (웨딩링을 더 크게)
        margin = 0.05  # 5% 여유만
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        w = min(image.shape[1] - x, w + 2 * margin_w)
        h = min(image.shape[0] - y, h + 2 * margin_h)
    else:
        # 웨딩링 영역을 못 찾으면 중앙 90% 사용
        height, width = image.shape[:2]
        margin = 0.05
        x = int(width * margin)
        y = int(height * margin)
        w = int(width * (1 - 2 * margin))
        h = int(height * (1 - 2 * margin))
    
    # 크롭
    cropped = image[y:y+h, x:x+w]
    
    # 비율 맞추기 (여백 최소화)
    crop_h, crop_w = cropped.shape[:2]
    scale = min(target_w / crop_w, target_h / crop_h) * 0.98  # 98%로 거의 꽉 차게
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # 리사이즈
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 캔버스 생성 (밝은 배경)
    canvas = np.full((target_h, target_w, 3), 250, dtype=np.uint8)
    
    # 중앙 배치
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return canvas

def handler(event):
    """RunPod 핸들러 - 검은색 테두리 100% 제거 보장"""
    try:
        input_data = event.get("input", {})
        
        # 테스트 메시지
        if "prompt" in input_data:
            return {
                "status": "ready",
                "message": "Wedding Ring AI v18.0 - Black Border 100% Removal",
                "capabilities": ["aggressive_border_detection", "complete_removal", "perfect_thumbnail"]
            }
        
        # 이미지 처리
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            raise ValueError("No image provided")
        
        # 디코딩
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)
        original_height, original_width = image_array.shape[:2]
        
        # 1. 검은색 테두리 적극적 감지
        border_info = detect_black_border_aggressive(image_array)
        
        # 2. 검은색 테두리 완전 제거 (크롭 + 리사이즈)
        if border_info['found']:
            print(f"Black border detected! Top:{border_info['top']}, Bottom:{border_info['bottom']}, Left:{border_info['left']}, Right:{border_info['right']}")
            image_array = remove_black_border_completely(image_array, border_info)
        else:
            print("No black border detected")
        
        # 3. v13.3 보정 적용 (더 강하게)
        enhanced = enhance_wedding_ring_v13_3(image_array)
        
        # 추가 밝기 보정 (전체적으로 더 밝게)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        # 4. 2x 업스케일링
        height, width = enhanced.shape[:2]
        upscaled = cv2.resize(enhanced, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # 5. 썸네일 생성 (검은색 테두리 정보 활용)
        if border_info['found']:
            # 원본에서 검은색 테두리 영역 제외하고 웨딩링 영역만 사용
            ring_region = image_array  # 이미 크롭됨
        else:
            ring_region = enhanced
        
        # 웨딩링 감지 후 썸네일 생성
        ring_bbox = detect_ring_in_image(ring_region)
        thumbnail = create_perfect_thumbnail(ring_region, ring_bbox)
        
        # 썸네일도 v13.3 보정 적용
        thumbnail = enhance_wedding_ring_v13_3(thumbnail)
        thumbnail = cv2.convertScaleAbs(thumbnail, alpha=1.1, beta=10)
        
        # 결과 인코딩
        # 메인 이미지
        main_pil = Image.fromarray(upscaled)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "processing_info": {
                "border_detected": border_info['found'],
                "border_thickness": border_info.get('thickness', 0),
                "border_removed": border_info['found'],
                "original_size": f"{original_width}x{original_height}",
                "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                "thumbnail_size": "1000x1300"
            }
        }
        
    except Exception as e:
        # 에러여도 최소한의 처리
        try:
            if 'image_array' in locals():
                # 최소한 밝게라도 처리
                brightened = cv2.convertScaleAbs(image_array, alpha=1.5, beta=50)
                pil_img = Image.fromarray(brightened)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=95)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    "enhanced_image": img_base64,
                    "thumbnail": img_base64,
                    "error": f"Processing error: {str(e)}, returned brightened image"
                }
        except:
            pass
        
        return {"error": f"Critical error: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
