import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 - 색온도 조정 수정 (파란색 방지)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': 0,  # 색온도 조정 제거 (파란색 방지)
            'color_temp_b': 0,
            'original_blend': 0.15,
            'saturation': 1.02,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.18,
            'saturation': 1.00,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': 0,
            'color_temp_b': 0,
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
            'color_temp_a': 2,  # 로즈골드는 약간 따뜻하게
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
            'brightness': 1.30,  # 밝게 유지
            'contrast': 1.11,
            'white_overlay': 0.15,  # 하얀색 오버레이로 밝게
            'sharpness': 1.16,
            'color_temp_a': 0,  # 색온도 조정 제거
            'color_temp_b': 0,
            'original_blend': 0.15,
            'saturation': 0.90,
            'gamma': 1.00
        },
        'warm': {
            'brightness': 1.28,
            'contrast': 1.10,
            'white_overlay': 0.17,
            'sharpness': 1.14,
            'color_temp_a': 0,
            'color_temp_b': 0,
            'original_blend': 0.17,
            'saturation': 0.88,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.32,
            'contrast': 1.13,
            'white_overlay': 0.13,
            'sharpness': 1.18,
            'color_temp_a': 0,
            'color_temp_b': 0,
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
            'color_temp_a': 3,  # 옐로우골드는 따뜻하게
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

# 28쌍 AFTER 배경색
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [250, 248, 245],
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

def detect_black_border_guaranteed(image):
    """검은색 테두리 감지 - 확실하게 찾기"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 가장자리 영역만 확인 (더 넓게)
    edge_size = min(200, min(h, w) // 4)  # 최대 200픽셀 또는 이미지의 1/4
    
    # 가장자리 마스크 생성
    edge_mask = np.zeros_like(gray)
    edge_mask[:edge_size, :] = 255  # 상단
    edge_mask[-edge_size:, :] = 255  # 하단
    edge_mask[:, :edge_size] = 255  # 좌측
    edge_mask[:, -edge_size:] = 255  # 우측
    
    # 여러 threshold로 검은색 영역 찾기
    combined_mask = np.zeros_like(gray)
    
    for thresh in [10, 20, 30, 40, 50, 60, 70, 80]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # 가장자리에서만 검은색 찾기
        edge_binary = cv2.bitwise_and(binary, edge_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edge_binary)
    
    # 모든 가장자리에 검은색이 있는지 확인
    top_black = np.sum(combined_mask[:50, :]) > 0
    bottom_black = np.sum(combined_mask[-50:, :]) > 0
    left_black = np.sum(combined_mask[:, :50]) > 0
    right_black = np.sum(combined_mask[:, -50:]) > 0
    
    if top_black and bottom_black and left_black and right_black:
        # 검은색 테두리가 있다고 판단
        # 실제 테두리 두께 측정
        thickness_top = 0
        thickness_bottom = 0
        thickness_left = 0
        thickness_right = 0
        
        # 상단
        for i in range(edge_size):
            if np.mean(combined_mask[i, w//4:3*w//4]) < 128:
                break
            thickness_top = i
        
        # 하단
        for i in range(edge_size):
            if np.mean(combined_mask[h-i-1, w//4:3*w//4]) < 128:
                break
            thickness_bottom = i
        
        # 좌측
        for i in range(edge_size):
            if np.mean(combined_mask[h//4:3*h//4, i]) < 128:
                break
            thickness_left = i
        
        # 우측
        for i in range(edge_size):
            if np.mean(combined_mask[h//4:3*h//4, w-i-1]) < 128:
                break
            thickness_right = i
        
        # 평균 두께 (안전 마진 포함)
        avg_thickness = int((thickness_top + thickness_bottom + thickness_left + thickness_right) / 4 * 1.2) + 10
        
        return combined_mask, avg_thickness
    
    return None, None

def detect_metal_type_simple(image):
    """금속 타입 감지 - 단순화"""
    # 중앙 영역만 분석
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    size = min(h, w) // 4
    
    center_region = image[center_y-size:center_y+size, center_x-size:center_x+size]
    
    # 평균 색상
    avg_color = np.mean(center_region, axis=(0, 1))
    avg_b, avg_g, avg_r = avg_color
    
    # HSV로 변환
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])
    
    # 밝기와 채도로 판단
    if avg_sat < 30 and avg_val > 180:
        return 'white_gold'
    elif avg_hue < 20 and avg_sat > 50:
        return 'rose_gold'
    elif 15 <= avg_hue <= 35 and avg_sat < 50:
        return 'champagne_gold'
    elif 20 <= avg_hue <= 40:
        return 'yellow_gold'
    else:
        return 'white_gold'

def apply_v13_enhancement_safe(image, params):
    """v13.3 보정 - 안전하게 (파란색 방지)"""
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
    hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * params.get('saturation', 1.0)
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    enhanced_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 6. 하얀색 오버레이
    white_overlay = np.full_like(enhanced_array, 255)
    enhanced_array = cv2.addWeighted(
        enhanced_array, 1 - params['white_overlay'],
        white_overlay, params['white_overlay'], 0
    )
    
    # 7. 색온도 조정 (약하게만)
    if params['color_temp_a'] != 0 or params['color_temp_b'] != 0:
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE
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

def remove_border_completely(image, thickness):
    """검은색 테두리 완전 제거"""
    h, w = image.shape[:2]
    
    # 배경색 (깨끗한 흰색에 가까운 색)
    bg_color = [250, 248, 245]
    
    # 결과 이미지
    result = image.copy()
    
    # 테두리 영역을 배경색으로 직접 교체
    # 상단
    result[:thickness, :] = bg_color
    # 하단
    result[-thickness:, :] = bg_color
    # 좌측
    result[:, :thickness] = bg_color
    # 우측
    result[:, -thickness:] = bg_color
    
    # 모서리 부분 추가 처리
    corner_size = int(thickness * 1.5)
    # 좌상단
    result[:corner_size, :corner_size] = bg_color
    # 우상단
    result[:corner_size, -corner_size:] = bg_color
    # 좌하단
    result[-corner_size:, :corner_size] = bg_color
    # 우하단
    result[-corner_size:, -corner_size:] = bg_color
    
    # 웨딩링 영역 (테두리 안쪽)
    ring_region = image[thickness:-thickness, thickness:-thickness].copy()
    
    return result, ring_region, (thickness, thickness, w-2*thickness, h-2*thickness)

def create_thumbnail_exact(image, target_size=(1000, 1300)):
    """정확한 1000×1300 썸네일"""
    h, w = image.shape[:2]
    
    # 중앙 영역 추출
    if w / h > 1000 / 1300:  # 이미지가 더 넓은 경우
        new_w = int(h * 1000 / 1300)
        x = (w - new_w) // 2
        cropped = image[:, x:x+new_w]
    else:  # 이미지가 더 높은 경우
        new_h = int(w * 1300 / 1000)
        y = (h - new_h) // 2
        cropped = image[y:y+new_h, :]
    
    # 정확히 1000×1300으로 리사이즈
    thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """RunPod 핸들러 v24.0"""
    try:
        input_data = event["input"]
        
        # 테스트 요청
        if "prompt" in input_data:
            return {
                "output": {
                    "message": "Wedding Ring AI v24.0 - Final Fix",
                    "status": "ready"
                }
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # 검은색 테두리 감지
            border_mask, thickness = detect_black_border_guaranteed(image_array)
            
            if border_mask is not None and thickness > 0:
                # 테두리가 있는 경우
                # 1. 테두리 완전 제거
                result_image, ring_region, ring_bbox = remove_border_completely(image_array, thickness)
                
                # 2. 금속 타입 감지 (웨딩링 영역에서)
                metal_type = detect_metal_type_simple(ring_region)
                lighting = 'natural'  # 단순화
                
                # 3. 웨딩링 영역만 v13.3 보정
                params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                    WEDDING_RING_PARAMS['white_gold']['natural'])
                enhanced_ring = apply_v13_enhancement_safe(ring_region, params)
                
                # 4. 보정된 웨딩링을 결과 이미지에 넣기
                rx, ry, rw, rh = ring_bbox
                result_image[ry:ry+rh, rx:rx+rw] = enhanced_ring
                
                # 5. 경계 부분 부드럽게
                # 간단한 블러로 경계 처리
                mask = np.zeros(result_image.shape[:2], dtype=np.float32)
                cv2.rectangle(mask, (rx+10, ry+10), (rx+rw-10, ry+rh-10), 1.0, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), 10)
                
                # 블렌딩
                original_with_border_removed = result_image.copy()
                for c in range(3):
                    result_image[:, :, c] = (result_image[:, :, c] * mask + 
                                           original_with_border_removed[:, :, c] * (1 - mask)).astype(np.uint8)
                
            else:
                # 테두리가 없는 경우 - 전체 보정
                metal_type = detect_metal_type_simple(image_array)
                lighting = 'natural'
                
                params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                    WEDDING_RING_PARAMS['white_gold']['natural'])
                result_image = apply_v13_enhancement_safe(image_array, params)
            
            # 2x 업스케일링
            upscaled = cv2.resize(result_image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            
            # 썸네일 생성
            thumbnail = create_thumbnail_exact(upscaled)
            
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
                        "metal_type": metal_type,
                        "border_detected": border_mask is not None,
                        "border_thickness": thickness if thickness else 0,
                        "version": "v24.0"
                    }
                }
            }
            
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "message": "처리 중 오류가 발생했습니다."
            }
        }

# RunPod 시작
runpod.serverless.start({"handler": handler})
