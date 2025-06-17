import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반 - 4금속 x 3조명 = 12가지)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15,
            'saturation': 1.02,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18,
            'saturation': 1.00,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12,
            'saturation': 1.03,
            'gamma': 1.02
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15,
            'contrast': 1.08,
            'white_overlay': 0.06,
            'sharpness': 1.15,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.20,
            'saturation': 1.10,
            'gamma': 0.98
        },
        'warm': {
            'brightness': 1.13,
            'contrast': 1.06,
            'white_overlay': 0.08,
            'sharpness': 1.12,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.22,
            'saturation': 1.08,
            'gamma': 0.96
        },
        'cool': {
            'brightness': 1.17,
            'contrast': 1.10,
            'white_overlay': 0.05,
            'sharpness': 1.17,
            'color_temp_a': 3,
            'color_temp_b': 2,
            'original_blend': 0.18,
            'saturation': 1.12,
            'gamma': 1.00
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.30,  # 대화 25번: 화이트골드처럼 밝게
            'contrast': 1.11,
            'white_overlay': 0.15,  # 87% 증가
            'sharpness': 1.16,
            'color_temp_a': -6,  # 화이트 방향
            'color_temp_b': -6,
            'original_blend': 0.15,
            'saturation': 0.90,  # 채도 감소
            'gamma': 1.00
        },
        'warm': {
            'brightness': 1.28,
            'contrast': 1.10,
            'white_overlay': 0.17,
            'sharpness': 1.14,
            'color_temp_a': -7,
            'color_temp_b': -7,
            'original_blend': 0.17,
            'saturation': 0.88,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.32,
            'contrast': 1.13,
            'white_overlay': 0.13,
            'sharpness': 1.18,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.13,
            'saturation': 0.92,
            'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16,
            'contrast': 1.09,
            'white_overlay': 0.05,
            'sharpness': 1.14,
            'color_temp_a': 3,
            'color_temp_b': 2,
            'original_blend': 0.22,
            'saturation': 1.15,
            'gamma': 0.97
        },
        'warm': {
            'brightness': 1.14,
            'contrast': 1.07,
            'white_overlay': 0.07,
            'sharpness': 1.12,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.25,
            'saturation': 1.12,
            'gamma': 0.95
        },
        'cool': {
            'brightness': 1.18,
            'contrast': 1.11,
            'white_overlay': 0.04,
            'sharpness': 1.16,
            'color_temp_a': 4,
            'color_temp_b': 3,
            'original_blend': 0.20,
            'saturation': 1.18,
            'gamma': 0.99
        }
    }
}

# 대화 28번: 28쌍 AFTER 배경색
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [250, 248, 245],   # 밝고 깨끗한 배경
        'medium': [245, 242, 238],
        'default': [250, 248, 245]
    },
    'warm': {
        'light': [252, 248, 242],
        'medium': [248, 243, 235],
        'default': [252, 248, 242]
    },
    'cool': {
        'light': [245, 248, 252],
        'medium': [238, 242, 248],
        'default': [245, 248, 252]
    }
}

def detect_black_border_ultra_strong(image):
    """대화 29번: 적응형 검은색 테두리 감지 - 100픽셀 두께까지 대응"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 전체 이미지에서 검은색 찾기 (threshold 5~50)
    combined_mask = np.zeros_like(gray)
    
    for thresh in range(5, 51, 5):
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 사각형 찾기
        largest_rect = None
        max_area = 0
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            
            # 사각형이 이미지의 50% 이상이고 테두리를 포함하는지 확인
            if area > max_area and area > (w * h * 0.5):
                if x <= 50 and y <= 50 and (x + cw) >= (w - 50) and (y + ch) >= (h - 50):
                    largest_rect = (x, y, cw, ch)
                    max_area = area
        
        if largest_rect:
            return combined_mask, largest_rect
    
    return None, None

def detect_actual_line_thickness(mask, bbox):
    """대화 29번: 실제 검은색 선 두께 측정"""
    x, y, w, h = bbox
    
    # 4방향에서 실제 두께 측정
    top_thickness = 0
    bottom_thickness = 0
    left_thickness = 0
    right_thickness = 0
    
    # 상단 두께
    for i in range(min(200, h)):
        if np.mean(mask[y+i, x:x+w]) < 128:
            break
        top_thickness = i
    
    # 하단 두께
    for i in range(min(200, h)):
        if np.mean(mask[y+h-i-1, x:x+w]) < 128:
            break
        bottom_thickness = i
    
    # 좌측 두께
    for i in range(min(200, w)):
        if np.mean(mask[y:y+h, x+i]) < 128:
            break
        left_thickness = i
    
    # 우측 두께
    for i in range(min(200, w)):
        if np.mean(mask[y:y+h, x+w-i-1]) < 128:
            break
        right_thickness = i
    
    # 최대값 × 1.5 + 안전 마진
    max_thickness = max(top_thickness, bottom_thickness, left_thickness, right_thickness)
    return int(max_thickness * 1.5) + 20

def detect_metal_type_advanced(image, mask=None):
    """금속 타입 감지 - 샴페인골드 우선"""
    # 마스크가 있으면 웨딩링 영역만 분석
    if mask is not None and np.any(mask):
        masked = cv2.bitwise_and(image, image, mask=mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
        
        # 마스크 영역의 평균 색상
        mask_pixels = np.where(mask > 0)
        if len(mask_pixels[0]) > 0:
            avg_hue = np.mean(hsv[mask_pixels[0], mask_pixels[1], 0])
            avg_sat = np.mean(hsv[mask_pixels[0], mask_pixels[1], 1])
            avg_val = np.mean(hsv[mask_pixels[0], mask_pixels[1], 2])
        else:
            return 'champagne_gold'
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
    
    # 샴페인골드 감지 우선
    if 15 <= avg_hue <= 35 and avg_sat < 50 and avg_val > 150:
        return 'champagne_gold'
    elif avg_sat < 30:
        return 'white_gold'
    elif avg_hue < 15 or avg_hue > 165:
        return 'rose_gold'
    elif 15 <= avg_hue <= 35:
        return 'yellow_gold'
    else:
        return 'white_gold'

def detect_lighting_condition(image):
    """조명 조건 감지"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_mean = np.mean(lab[:, :, 0])
    b_mean = np.mean(lab[:, :, 2])
    
    if b_mean < 125:
        return 'warm'
    elif b_mean > 135:
        return 'cool'
    else:
        return 'natural'

def apply_v13_complete_enhancement(image, params):
    """대화 16-20: v13.3 완전한 10단계 보정"""
    # 1. 노이즈 제거
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # PIL 이미지로 변환
    pil_image = Image.fromarray(denoised)
    
    # 2. 밝기 조정
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = brightness_enhancer.enhance(params['brightness'])
    
    # 3. 대비 조정
    contrast_enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = contrast_enhancer.enhance(params['contrast'])
    
    # 4. 선명도 조정
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness_enhancer.enhance(params['sharpness'])
    
    # numpy 배열로 변환
    enhanced_array = np.array(enhanced)
    
    # 5. 채도 조정
    hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.get('saturation', 1.0), 0, 255)
    enhanced_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 6. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
    white_overlay = np.full_like(enhanced_array, 255)
    enhanced_array = cv2.addWeighted(
        enhanced_array, 1 - params['white_overlay'],
        white_overlay, params['white_overlay'], 0
    )
    
    # 7. 색온도 조정 (LAB 색공간)
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE (명료도)
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 9. 감마 보정
    gamma = params.get('gamma', 1.0)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_array = cv2.LUT(enhanced_array, table)
    
    # 10. 원본과 블렌딩
    final = cv2.addWeighted(
        enhanced_array, 1 - params['original_blend'],
        image, params['original_blend'], 0
    )
    
    return final

def remove_border_and_enhance(image, border_mask, border_bbox, thickness, metal_type, lighting):
    """검은색 테두리 제거하고 웨딩링만 보정"""
    h, w = image.shape[:2]
    x, y, bw, bh = border_bbox
    
    # 웨딩링 영역 계산 (테두리 안쪽)
    ring_x = x + thickness
    ring_y = y + thickness
    ring_w = bw - (thickness * 2)
    ring_h = bh - (thickness * 2)
    
    # 웨딩링 영역이 너무 작으면 조정
    if ring_w < 100 or ring_h < 100:
        thickness = max(20, thickness // 2)
        ring_x = x + thickness
        ring_y = y + thickness
        ring_w = bw - (thickness * 2)
        ring_h = bh - (thickness * 2)
    
    # 결과 이미지 초기화
    result = image.copy()
    
    # 1. 테두리 영역을 AFTER 배경색으로 교체
    bg_color = AFTER_BACKGROUND_COLORS[lighting]['default']
    
    # 테두리 마스크 생성
    border_only_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(border_only_mask, (x, y), (x+bw, y+bh), 255, -1)
    cv2.rectangle(border_only_mask, (ring_x, ring_y), (ring_x+ring_w, ring_y+ring_h), 0, -1)
    
    # 배경색으로 교체
    result[border_only_mask > 0] = bg_color
    
    # 2. 웨딩링 영역만 추출해서 보정
    ring_region = image[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w].copy()
    
    # v13.3 파라미터로 웨딩링만 보정
    params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['white_gold']['natural'])
    enhanced_ring = apply_v13_complete_enhancement(ring_region, params)
    
    # 웨딩링 영역 추가 밝기 보정 (웨딩링만 더 밝게)
    enhanced_ring = cv2.convertScaleAbs(enhanced_ring, alpha=1.1, beta=10)
    
    # 3. 보정된 웨딩링을 결과 이미지에 넣기
    result[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w] = enhanced_ring
    
    # 4. 대화 31번: 31×31 가우시안 블러로 자연스러운 블렌딩
    # 경계 부분만 블러 처리
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask, (ring_x-10, ring_y-10), (ring_x+ring_w+10, ring_y+ring_h+10), 1.0, 20)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    
    # 블렌딩
    for c in range(3):
        result[:, :, c] = (result[:, :, c] * mask + image[:, :, c] * (1 - mask)).astype(np.uint8)
    
    return result, (ring_x, ring_y, ring_w, ring_h)

def create_perfect_thumbnail_1000x1300(image, ring_bbox):
    """정확한 1000×1300 썸네일 생성 - 웨딩링이 화면 가득"""
    if ring_bbox is None:
        # 웨딩링 영역이 없으면 중앙 기준
        h, w = image.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        ring_bbox = (x, y, size, size)
    
    x, y, rw, rh = ring_bbox
    
    # 웨딩링 중심점
    center_x = x + rw // 2
    center_y = y + rh // 2
    
    # 1000×1300 비율에 맞춰 크롭 영역 계산
    # 웨딩링이 화면의 80%를 차지하도록
    crop_size = int(max(rw, rh) / 0.8)
    
    # 1000:1300 = 1:1.3 비율
    crop_w = crop_size
    crop_h = int(crop_size * 1.3)
    
    # 크롭 시작점 (웨딩링이 중앙에 오도록)
    crop_x = max(0, center_x - crop_w // 2)
    crop_y = max(0, center_y - crop_h // 2)
    
    # 이미지 경계 체크
    if crop_x + crop_w > image.shape[1]:
        crop_x = image.shape[1] - crop_w
    if crop_y + crop_h > image.shape[0]:
        crop_y = image.shape[0] - crop_h
    
    # 크롭
    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # 정확히 1000×1300으로 리사이즈
    thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """RunPod 핸들러 - 42개 대화 완전체"""
    try:
        input_data = event["input"]
        
        # 테스트 요청 처리
        if "prompt" in input_data:
            return {
                "output": {
                    "message": "Wedding Ring AI v23.0 - 42개 대화 완전체",
                    "status": "ready"
                }
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 원본 이미지 백업
            original_image = image_array.copy()
            
            # 1. 검은색 테두리 감지
            border_mask, border_bbox = detect_black_border_ultra_strong(image_array)
            
            result_image = None
            thumbnail = None
            ring_bbox = None
            
            if border_mask is not None and border_bbox is not None:
                # 테두리가 있는 경우
                # 실제 두께 측정
                thickness = detect_actual_line_thickness(border_mask, border_bbox)
                
                # 금속 타입과 조명 감지
                metal_type = detect_metal_type_advanced(image_array, border_mask)
                lighting = detect_lighting_condition(image_array)
                
                # 테두리 제거하고 웨딩링만 보정
                result_image, ring_bbox = remove_border_and_enhance(
                    image_array, border_mask, border_bbox, thickness, metal_type, lighting
                )
                
            else:
                # 테두리가 없는 경우 - 전체 이미지 보정
                metal_type = detect_metal_type_advanced(image_array)
                lighting = detect_lighting_condition(image_array)
                
                params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                    WEDDING_RING_PARAMS['white_gold']['natural'])
                
                result_image = apply_v13_complete_enhancement(image_array, params)
                
                # 전체 이미지 기준으로 썸네일용 영역 설정
                h, w = result_image.shape[:2]
                ring_bbox = (w//4, h//4, w//2, h//2)
            
            # 2x 업스케일링
            upscaled = cv2.resize(result_image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            
            # 썸네일 생성 (1000×1300)
            if ring_bbox:
                # 업스케일링된 좌표로 변환
                scaled_ring_bbox = tuple(x * 2 for x in ring_bbox)
                thumbnail = create_perfect_thumbnail_1000x1300(upscaled, scaled_ring_bbox)
            else:
                thumbnail = create_perfect_thumbnail_1000x1300(upscaled, None)
            
            # 결과 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "output": {
                    "enhanced_image": main_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "metal_type": metal_type if 'metal_type' in locals() else "unknown",
                        "lighting": lighting if 'lighting' in locals() else "unknown",
                        "border_detected": border_bbox is not None,
                        "border_thickness": thickness if border_bbox else 0,
                        "thumbnail_size": "1000x1300"
                    }
                }
            }
            
    except Exception as e:
        # 에러 발생 시에도 결과 반환
        return {
            "output": {
                "error": str(e),
                "message": "처리 중 오류가 발생했습니다."
            }
        }

# RunPod 시작
runpod.serverless.start({"handler": handler})
