import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반) - 12가지 세트
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
            'saturation': 1.15,
            'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.05,
            'white_overlay': 0.05,
            'sharpness': 1.10,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.25,
            'saturation': 1.10,
            'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25,
            'contrast': 1.15,
            'white_overlay': 0.08,
            'sharpness': 1.25,
            'color_temp_a': 3,
            'color_temp_b': 2,
            'original_blend': 0.15,
            'saturation': 1.25,
            'gamma': 1.02
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17,
            'contrast': 1.11,
            'white_overlay': 0.12,
            'sharpness': 1.16,
            'color_temp_a': -4,
            'color_temp_b': -4,
            'original_blend': 0.15,
            'saturation': 1.02,
            'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15,
            'contrast': 1.10,
            'white_overlay': 0.10,
            'sharpness': 1.14,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18,
            'saturation': 1.00,
            'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22,
            'contrast': 1.15,
            'white_overlay': 0.14,
            'sharpness': 1.18,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.12,
            'saturation': 1.05,
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
            'saturation': 1.20,
            'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.08,
            'white_overlay': 0.04,
            'sharpness': 1.12,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.25,
            'saturation': 1.12,
            'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28,
            'contrast': 1.20,
            'white_overlay': 0.07,
            'sharpness': 1.25,
            'color_temp_a': 4,
            'color_temp_b': 3,
            'original_blend': 0.18,
            'saturation': 1.28,
            'gamma': 1.03
        }
    }
}

def detect_metal_type(image):
    """금속 타입 자동 감지"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]
    
    avg_hue = np.mean(center_region[:, :, 0])
    avg_sat = np.mean(center_region[:, :, 1])
    avg_val = np.mean(center_region[:, :, 2])
    
    if avg_sat < 20:
        return 'white_gold'
    elif 5 <= avg_hue <= 25:
        if avg_sat > 80:
            return 'yellow_gold'
        else:
            return 'champagne_gold'
    elif avg_hue < 5 or avg_hue > 170:
        return 'rose_gold'
    else:
        return 'white_gold'

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

def detect_black_border(image):
    """검은색 테두리 정확히 감지 - 매우 강력한 버전"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 가장자리 영역만 확인 (200픽셀)
    edge_mask = np.zeros_like(gray)
    edge_width = 200
    edge_mask[:edge_width, :] = 255
    edge_mask[-edge_width:, :] = 255
    edge_mask[:, :edge_width] = 255
    edge_mask[:, -edge_width:] = 255
    
    # 여러 threshold로 검은색 찾기 (확실하게)
    combined_mask = np.zeros_like(gray)
    for thresh in [15, 20, 25, 30, 35, 40, 45, 50]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_and(binary, edge_mask)
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 사각형 테두리 찾기
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    
    # 가장 큰 사각형 찾기
    best_bbox = None
    best_area = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # 이미지 크기의 30% 이상인 사각형만
        if area > (gray.shape[0] * gray.shape[1] * 0.3):
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)
    
    if best_bbox is None:
        return None, None, 0
    
    # 실제 선 두께 측정
    x, y, w, h = best_bbox
    thicknesses = []
    
    # 상단
    for i in range(min(100, h//2)):
        if np.mean(gray[y+i, x:x+w]) < 50:
            thicknesses.append(i)
        else:
            break
    
    # 하단
    for i in range(min(100, h//2)):
        if np.mean(gray[y+h-1-i, x:x+w]) < 50:
            thicknesses.append(i)
        else:
            break
    
    # 좌측
    for i in range(min(100, w//2)):
        if np.mean(gray[y:y+h, x+i]) < 50:
            thicknesses.append(i)
        else:
            break
    
    # 우측
    for i in range(min(100, w//2)):
        if np.mean(gray[y:y+h, x+w-1-i]) < 50:
            thicknesses.append(i)
        else:
            break
    
    if thicknesses:
        thickness = max(thicknesses) + 20  # 여유 있게
    else:
        thickness = 50
    
    return combined_mask, best_bbox, thickness

def remove_black_border(image, mask, bbox, thickness):
    """검은색 테두리 완전 제거"""
    if bbox is None:
        return image
    
    x, y, w, h = bbox
    result = image.copy()
    
    # 배경색 (깨끗한 흰색)
    bg_color = np.array([250, 248, 245])
    
    # 테두리 영역만 제거 (두께 + 여유)
    remove_thickness = thickness + 10
    
    # 상단
    result[:y+remove_thickness, :] = bg_color
    # 하단
    result[y+h-remove_thickness:, :] = bg_color
    # 좌측
    result[:, :x+remove_thickness] = bg_color
    # 우측
    result[:, x+w-remove_thickness:] = bg_color
    
    # 모서리 부분도 깔끔하게
    # 좌상단
    result[y:y+remove_thickness, x:x+remove_thickness] = bg_color
    # 우상단
    result[y:y+remove_thickness, x+w-remove_thickness:x+w] = bg_color
    # 좌하단
    result[y+h-remove_thickness:y+h, x:x+remove_thickness] = bg_color
    # 우하단
    result[y+h-remove_thickness:y+h, x+w-remove_thickness:x+w] = bg_color
    
    # 부드러운 블렌딩
    # 블렌딩 마스크 생성
    blend_mask = np.zeros(result.shape[:2], dtype=np.float32)
    blend_mask[:y+remove_thickness, :] = 1.0
    blend_mask[y+h-remove_thickness:, :] = 1.0
    blend_mask[:, :x+remove_thickness] = 1.0
    blend_mask[:, x+w-remove_thickness:] = 1.0
    
    # 가우시안 블러로 부드럽게
    blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 10)
    blend_mask = np.expand_dims(blend_mask, axis=2)
    
    # 블렌딩 적용
    bg_image = np.full_like(result, bg_color)
    result = (result * (1 - blend_mask) + bg_image * blend_mask).astype(np.uint8)
    
    return result

def apply_v13_enhancement(image, metal_type, lighting):
    """v13.3 완전한 10단계 보정"""
    params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['white_gold']['natural'])
    
    # 1. 노이즈 제거
    enhanced = cv2.bilateralFilter(image, 9, 75, 75)
    
    # 2-4. PIL로 기본 보정 (밝기, 대비, 선명도)
    pil_image = Image.fromarray(enhanced)
    
    # 밝기
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(params['brightness'])
    
    # 대비
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(params['contrast'])
    
    # 선명도
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(params['sharpness'])
    
    enhanced = np.array(pil_image)
    
    # 5. 채도 조정
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * params['saturation']
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 6. 하얀색 오버레이
    white_overlay = np.full_like(enhanced, 255)
    enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                              white_overlay, params['white_overlay'], 0)
    
    # 7. 색온도 조정 (LAB)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE (명료도)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 9. 감마 보정
    gamma = params['gamma']
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    # 10. 원본과 블렌딩
    enhanced = cv2.addWeighted(enhanced, 1 - params['original_blend'], 
                              image, params['original_blend'], 0)
    
    return enhanced

def create_perfect_thumbnail(image, bbox):
    """1000x1300 완벽한 썸네일 생성"""
    h, w = image.shape[:2]
    
    if bbox:
        x, y, bw, bh = bbox
        # 테두리 안쪽 영역 (여유 있게)
        margin = 50
        crop_x = x + margin
        crop_y = y + margin
        crop_w = bw - 2 * margin
        crop_h = bh - 2 * margin
        
        # 안전 체크
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(crop_w, w - crop_x)
        crop_h = min(crop_h, h - crop_y)
        
        if crop_w > 100 and crop_h > 100:
            cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        else:
            # 중앙 50% 크롭
            crop_w = w // 2
            crop_h = h // 2
            crop_x = w // 4
            crop_y = h // 4
            cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    else:
        # 테두리 없으면 중앙 크롭
        size = min(h, w) // 2
        center_x = w // 2
        center_y = h // 2
        crop_x = center_x - size // 2
        crop_y = center_y - size // 2
        cropped = image[crop_y:crop_y+size, crop_x:crop_x+size]
    
    # 1000x1300으로 리사이즈
    thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
    
    # 추가 선명도
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    thumbnail = cv2.filter2D(thumbnail, -1, kernel)
    
    return thumbnail

def handler(event):
    """RunPod 핸들러"""
    try:
        input_data = event["input"]
        
        # 테스트 메시지
        if "prompt" in input_data:
            return {
                "status": "v19.0 작동 중 - 검은색 테두리 확실히 제거",
                "handler": "handler.py 실행됨"
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
            
            # 2. v13.3 보정 적용
            enhanced = apply_v13_enhancement(image_array, metal_type, lighting)
            
            # 3. 검은색 테두리 감지 및 제거
            mask, bbox, thickness = detect_black_border(enhanced)
            if bbox and thickness > 10:
                enhanced = remove_black_border(enhanced, mask, bbox, thickness)
            
            # 4. 2x 업스케일링
            height, width = enhanced.shape[:2]
            upscaled = cv2.resize(enhanced, (width * 2, height * 2), 
                                 interpolation=cv2.INTER_LANCZOS4)
            
            # 5. 썸네일 생성
            thumbnail = create_perfect_thumbnail(enhanced, bbox)
            
            # 6. 결과 인코딩
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
                    "border_detected": bbox is not None,
                    "border_thickness": thickness if bbox else 0,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                    "thumbnail_size": "1000x1300"
                }
            }
            
    except Exception as e:
        return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
