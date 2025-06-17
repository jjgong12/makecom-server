import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# 28쌍 학습 데이터 기반 완전한 파라미터 (대화 3-15)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18, 'saturation': 1.00, 'gamma': 1.02
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.03, 'gamma': 0.99
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.06, 'white_overlay': 0.05,
            'sharpness': 1.12, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.22, 'saturation': 1.12, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.18, 'contrast': 1.10, 'white_overlay': 0.07,
            'sharpness': 1.18, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.18, 'saturation': 1.18, 'gamma': 1.00
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.14, 'contrast': 1.09, 'white_overlay': 0.10,
            'sharpness': 1.14, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.17, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.13, 'white_overlay': 0.14,
            'sharpness': 1.18, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.13, 'saturation': 1.04, 'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.13, 'contrast': 1.07, 'white_overlay': 0.04,
            'sharpness': 1.12, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.25, 'saturation': 1.18, 'gamma': 0.99
        },
        'cool': {
            'brightness': 1.19, 'contrast': 1.11, 'white_overlay': 0.06,
            'sharpness': 1.16, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.20, 'saturation': 1.22, 'gamma': 1.03
        }
    }
}

def detect_black_border(image):
    """검은색 테두리 감지 (대화 39번 - 더 강력한 감지)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 가장자리 영역 정의 (대화 34 - 가장자리만 확인)
    edge_width = min(200, min(h, w) // 4)  # 최대 200픽셀 또는 이미지의 1/4
    
    edge_mask = np.zeros_like(gray)
    edge_mask[:edge_width, :] = 255  # 상단
    edge_mask[-edge_width:, :] = 255  # 하단
    edge_mask[:, :edge_width] = 255  # 좌측
    edge_mask[:, -edge_width:] = 255  # 우측
    
    # 여러 threshold로 검은색 감지 (대화 38)
    combined_mask = np.zeros_like(gray)
    for thresh in [20, 30, 40, 50, 60, 70]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        border_only = cv2.bitwise_and(binary, edge_mask)
        combined_mask = cv2.bitwise_or(combined_mask, border_only)
    
    # 모폴로지 연산으로 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 가장 큰 사각형 컨투어 찾기
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 적응형 두께 감지 (대화 29)
        actual_thickness = detect_actual_line_thickness(combined_mask, (x, y, w, h))
        
        return True, (x, y, w, h), actual_thickness, combined_mask
    
    return False, None, 0, None

def detect_actual_line_thickness(mask, bbox):
    """실제 검은색 선 두께 측정 (대화 29번 핵심)"""
    x, y, w, h = bbox
    
    # 4방향에서 실제 두께 측정
    thicknesses = []
    
    # 상단
    for i in range(min(150, h//2)):
        if np.mean(mask[y+i, x:x+w]) < 128:
            break
        thicknesses.append(i)
    
    # 하단
    for i in range(min(150, h//2)):
        if np.mean(mask[y+h-i-1, x:x+w]) < 128:
            break
        thicknesses.append(i)
    
    # 좌측
    for i in range(min(150, w//2)):
        if np.mean(mask[y:y+h, x+i]) < 128:
            break
        thicknesses.append(i)
    
    # 우측
    for i in range(min(150, w//2)):
        if np.mean(mask[y:y+h, x+w-i-1]) < 128:
            break
        thicknesses.append(i)
    
    if thicknesses:
        # 안전하게 50% 더 추가
        return int(np.max(thicknesses) * 1.5)
    return 50

def detect_metal_type(image):
    """금속 타입 감지"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    
    if avg_sat < 30:
        return 'white_gold'
    elif 5 <= avg_hue <= 25:
        return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
    elif avg_hue < 5 or avg_hue > 170:
        return 'rose_gold'
    else:
        return 'champagne_gold'  # 기본값

def detect_lighting(image):
    """조명 환경 감지"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    b_mean = np.mean(b_channel)
    
    if b_mean < 125:
        return 'warm'
    elif b_mean > 135:
        return 'cool'
    else:
        return 'natural'

def enhance_wedding_ring_v13_3(image, metal_type, lighting):
    """v13.3 완전한 10단계 보정 (대화 16-20)"""
    params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['white_gold']['natural'])
    
    # 1. 노이즈 제거
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # PIL로 변환
    pil_image = Image.fromarray(denoised)
    
    # 2. 밝기 조정
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(params['brightness'])
    
    # 3. 대비 조정
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(params['contrast'])
    
    # 4. 선명도 조정
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(params['sharpness'])
    
    # numpy로 변환
    enhanced_array = np.array(enhanced)
    
    # 5. 채도 조정
    hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= params.get('saturation', 1.05)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    enhanced_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 6. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
    white_overlay = np.full_like(enhanced_array, 255)
    enhanced_array = cv2.addWeighted(enhanced_array, 1 - params['white_overlay'], 
                                   white_overlay, params['white_overlay'], 0)
    
    # 7. LAB 색온도 조정
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
    enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE (명료도)
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 9. 감마 보정
    gamma = params.get('gamma', 1.0)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
    enhanced_array = cv2.LUT(enhanced_array, table)
    
    # 10. 원본과 블렌딩
    result = cv2.addWeighted(enhanced_array, 1 - params['original_blend'], 
                           image, params['original_blend'], 0)
    
    return result

def remove_black_border(image, border_bbox, thickness):
    """검은색 테두리 제거 (대화 27-28 방식)"""
    x, y, w, h = border_bbox
    result = image.copy()
    
    # 배경색 (28쌍 AFTER 기준)
    bg_color = np.array([250, 248, 245])
    
    # 가장자리 두께만큼 배경색으로 채우기
    # 상단
    if y + thickness < result.shape[0]:
        result[:y+thickness, :] = bg_color
    
    # 하단
    if y + h - thickness >= 0:
        result[max(0, y+h-thickness):, :] = bg_color
    
    # 좌측
    if x + thickness < result.shape[1]:
        result[:, :x+thickness] = bg_color
    
    # 우측
    if x + w - thickness >= 0:
        result[:, max(0, x+w-thickness):] = bg_color
    
    # 부드러운 블렌딩 (대화 25)
    # 경계 부분에 가우시안 블러 적용
    mask = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x+thickness, y+thickness), 
                 (x+w-thickness, y+h-thickness), 255, -1)
    
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 10)
    mask_blur = mask_blur.astype(np.float32) / 255.0
    
    for c in range(3):
        result[:, :, c] = (image[:, :, c] * mask_blur + 
                          result[:, :, c] * (1 - mask_blur)).astype(np.uint8)
    
    return result

def create_perfect_thumbnail(image, border_bbox=None):
    """완벽한 1000x1300 썸네일 생성"""
    h, w = image.shape[:2]
    
    if border_bbox:
        x, y, w_box, h_box = border_bbox
        # 테두리 안쪽 웨딩링 영역에서 크롭
        margin = 0.1  # 10% 마진
        crop_x = int(x + w_box * margin)
        crop_y = int(y + h_box * margin)
        crop_w = int(w_box * (1 - 2 * margin))
        crop_h = int(h_box * (1 - 2 * margin))
        
        # 안전 체크
        crop_x = max(0, min(crop_x, w-1))
        crop_y = max(0, min(crop_y, h-1))
        crop_w = min(crop_w, w - crop_x)
        crop_h = min(crop_h, h - crop_y)
        
        if crop_w > 50 and crop_h > 50:
            cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        else:
            # 중앙 영역 사용
            size = int(min(h, w) * 0.8)
            center_x, center_y = w // 2, h // 2
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            cropped = image[y1:y1+size, x1:x1+size]
    else:
        # 테두리 없으면 중앙 80% 크롭
        size = int(min(h, w) * 0.8)
        center_x, center_y = w // 2, h // 2
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        cropped = image[y1:y1+size, x1:x1+size]
    
    # 1000x1300으로 리사이즈
    thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v19.0 - 39개 대화 완전 통합",
                "status": "ready"
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 1. 금속 타입 및 조명 감지
            metal_type = detect_metal_type(image_array)
            lighting = detect_lighting(image_array)
            
            # 2. v13.3 완전 보정 적용
            enhanced = enhance_wedding_ring_v13_3(image_array, metal_type, lighting)
            
            # 3. 검은색 테두리 감지
            has_border, border_bbox, thickness, mask = detect_black_border(enhanced)
            
            # 4. 테두리 제거
            if has_border and border_bbox and thickness > 0:
                final_image = remove_black_border(enhanced, border_bbox, thickness)
            else:
                final_image = enhanced
            
            # 5. 2x 업스케일링
            height, width = final_image.shape[:2]
            upscaled = cv2.resize(final_image, (width * 2, height * 2), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            # 6. 썸네일 생성
            if has_border and border_bbox:
                # 업스케일된 좌표로 변환
                scaled_bbox = (border_bbox[0] * 2, border_bbox[1] * 2, 
                             border_bbox[2] * 2, border_bbox[3] * 2)
                thumbnail = create_perfect_thumbnail(upscaled, scaled_bbox)
            else:
                thumbnail = create_perfect_thumbnail(upscaled)
            
            # 7. 결과 인코딩
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
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": has_border,
                    "border_thickness": thickness if has_border else 0,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "version": "v19.0 - 39 dialogs complete"
                }
            }
            
    except Exception as e:
        # 에러여도 최소한의 처리
        try:
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=20)
            upscaled = cv2.resize(enhanced, (enhanced.shape[1] * 2, enhanced.shape[0] * 2))
            thumbnail = cv2.resize(enhanced, (1000, 1300))
            
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {"error": str(e), "fallback": True}
            }
        except:
            return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
