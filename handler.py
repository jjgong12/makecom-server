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
            'brightness': 1.30,  # 더 밝게 (대화 25번: 화이트골드 방향)
            'contrast': 1.15,
            'white_overlay': 0.15,  # 대화 25번: 87% 증가
            'sharpness': 1.20,
            'color_temp_a': -6,  # 대화 25번: -1 → -6
            'color_temp_b': -6,  # 화이트에 가깝게
            'original_blend': 0.10,
            'saturation': 0.90,  # 채도 대폭 감소
            'gamma': 1.05
        },
        'warm': {
            'brightness': 1.28,
            'contrast': 1.14,
            'white_overlay': 0.13,
            'sharpness': 1.18,
            'color_temp_a': -7,
            'color_temp_b': -7,
            'original_blend': 0.12,
            'saturation': 0.88,
            'gamma': 1.03
        },
        'cool': {
            'brightness': 1.35,
            'contrast': 1.18,
            'white_overlay': 0.17,
            'sharpness': 1.22,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.08,
            'saturation': 0.92,
            'gamma': 1.06
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
    
    # 채도가 낮으면 화이트골드 또는 샴페인골드
    if avg_sat < 25:
        # 밝기로 구분
        if avg_val > 180:
            return 'white_gold'
        else:
            return 'champagne_gold'  # 무도금 화이트골드
    elif 5 <= avg_hue <= 25:
        if avg_sat > 80:
            return 'yellow_gold'
        else:
            return 'champagne_gold'
    elif avg_hue < 5 or avg_hue > 170:
        return 'rose_gold'
    else:
        return 'champagne_gold'  # 기본값을 샴페인골드로

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

def detect_black_border_v20(image):
    """검은색 테두리 감지 - 대화 29번 적응형 두께 감지"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # 전체 이미지에서 검은색 찾기 (가장자리 제한 없이)
    combined_mask = np.zeros_like(gray)
    
    # 여러 threshold로 검은색 찾기 (10-50)
    for thresh in range(10, 51, 5):
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    
    # 형태학적 연산으로 정리
    kernel = np.ones((10, 10), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    
    # 가장 큰 사각형 찾기
    best_bbox = None
    best_area = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # 이미지 크기의 50% 이상인 사각형
        if area > (gray.shape[0] * gray.shape[1] * 0.5):
            # 중앙에 가까운 사각형 우선
            center_x = x + w // 2
            center_y = y + h // 2
            dist_to_center = abs(center_x - gray.shape[1]//2) + abs(center_y - gray.shape[0]//2)
            
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)
    
    if best_bbox is None:
        return None, None, 0
    
    # 실제 선 두께 측정 - 대화 29번 방식
    x, y, w, h = best_bbox
    thickness = detect_actual_line_thickness(gray, best_bbox)
    
    return combined_mask, best_bbox, thickness

def detect_actual_line_thickness(gray, bbox):
    """대화 29번: 실제 선 두께 측정"""
    x, y, w, h = bbox
    thicknesses = []
    
    # 각 방향에서 실제 두께 측정
    # 상단
    for i in range(min(150, h//2)):
        if np.mean(gray[y+i, x+50:x+w-50]) < 40:
            continue
        else:
            thicknesses.append(i)
            break
    
    # 하단
    for i in range(min(150, h//2)):
        if np.mean(gray[y+h-1-i, x+50:x+w-50]) < 40:
            continue
        else:
            thicknesses.append(i)
            break
    
    # 좌측
    for i in range(min(150, w//2)):
        if np.mean(gray[y+50:y+h-50, x+i]) < 40:
            continue
        else:
            thicknesses.append(i)
            break
    
    # 우측
    for i in range(min(150, w//2)):
        if np.mean(gray[y+50:y+h-50, x+w-1-i]) < 40:
            continue
        else:
            thicknesses.append(i)
            break
    
    if thicknesses:
        # 최대값 사용 + 50% 안전 마진
        thickness = int(max(thicknesses) * 1.5) + 20
    else:
        thickness = 100  # 기본값 100픽셀
    
    return thickness

def remove_black_border_v20(image, mask, bbox, thickness):
    """검은색 테두리 완전 제거 - 대화 27번 방식"""
    if bbox is None:
        return image
    
    x, y, w, h = bbox
    result = image.copy()
    
    # 28쌍 AFTER 배경색 (대화 28번)
    bg_color = np.array([252, 250, 248])  # 밝고 깨끗한 배경
    
    # 테두리 제거 (두께 기반)
    # 대화 27번: 배경색 직접 덮어쓰기
    
    # 상하좌우 테두리만 제거
    # 상단
    result[:y+thickness, :] = bg_color
    # 하단
    result[y+h-thickness:, :] = bg_color
    # 좌측
    result[:, :x+thickness] = bg_color
    # 우측
    result[:, x+w-thickness:] = bg_color
    
    # 대화 25번: 31×31 가우시안 블러로 자연스러운 블렌딩
    # 블렌딩 마스크 생성
    blend_mask = np.zeros(result.shape[:2], dtype=np.float32)
    blend_mask[:y+thickness, :] = 1.0
    blend_mask[y+h-thickness:, :] = 1.0
    blend_mask[:, :x+thickness] = 1.0
    blend_mask[:, x+w-thickness:] = 1.0
    
    # 가우시안 블러로 부드럽게
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 10)
    blend_mask = np.expand_dims(blend_mask, axis=2)
    
    # 블렌딩 적용
    bg_image = np.full_like(result, bg_color)
    result = (result * (1 - blend_mask) + bg_image * blend_mask).astype(np.uint8)
    
    return result

def apply_v13_enhancement(image, metal_type, lighting):
    """v13.3 완전한 10단계 보정 - 대화 16-20번"""
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
    
    # 6. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
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
    
    # 샴페인골드 추가 밝기 보정
    if metal_type == 'champagne_gold':
        # 전체적으로 더 밝게
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=10)
    
    return enhanced

def extract_ring_area(image, bbox, thickness):
    """웨딩링 영역만 추출 - 대화 34번 가장자리 보호"""
    if bbox is None:
        return None
    
    x, y, w, h = bbox
    
    # 테두리 안쪽 영역 (보수적으로)
    inner_margin = thickness + 30
    inner_x = x + inner_margin
    inner_y = y + inner_margin
    inner_w = w - 2 * inner_margin
    inner_h = h - 2 * inner_margin
    
    # 안전 체크
    if inner_w <= 100 or inner_h <= 100:
        # 중앙 50% 영역
        inner_x = x + w // 4
        inner_y = y + h // 4
        inner_w = w // 2
        inner_h = h // 2
    
    # 경계 체크
    inner_x = max(0, inner_x)
    inner_y = max(0, inner_y)
    inner_w = min(inner_w, image.shape[1] - inner_x)
    inner_h = min(inner_h, image.shape[0] - inner_y)
    
    return (inner_x, inner_y, inner_w, inner_h)

def create_perfect_thumbnail(image, bbox, ring_area):
    """1000x1300 완벽한 썸네일 - 대화 33번"""
    h, w = image.shape[:2]
    
    if ring_area:
        rx, ry, rw, rh = ring_area
        # 웨딩링 영역 기준으로 크롭
        cropped = image[ry:ry+rh, rx:rx+rw]
    elif bbox:
        x, y, bw, bh = bbox
        # 테두리 안쪽 크롭
        margin = 100
        crop_x = x + margin
        crop_y = y + margin
        crop_w = bw - 2 * margin
        crop_h = bh - 2 * margin
        cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    else:
        # 중앙 크롭
        size = min(h, w) // 2
        center_x = w // 2
        center_y = h // 2
        crop_x = center_x - size // 2
        crop_y = center_y - size // 2
        cropped = image[crop_y:crop_y+size, crop_x:crop_x+size]
    
    # 1000x1300으로 리사이즈
    thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
    
    # 추가 보정 (더 밝고 선명하게)
    thumbnail = cv2.convertScaleAbs(thumbnail, alpha=1.1, beta=5)
    
    # 언샤프 마스킹
    gaussian = cv2.GaussianBlur(thumbnail, (5, 5), 1.0)
    thumbnail = cv2.addWeighted(thumbnail, 1.5, gaussian, -0.5, 0)
    
    return thumbnail

def handler(event):
    """RunPod 핸들러 - v20.0 Final"""
    try:
        input_data = event["input"]
        
        # 테스트 메시지
        if "prompt" in input_data:
            return {
                "status": "v20.0 Final - 검은색 선 완전 제거 + 샴페인골드 화이트화",
                "handler": "handler.py 실행됨",
                "version": "대화 1-40번 모든 성과 포함"
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
            
            # 2. v13.3 보정 적용 (검은색 선 제거 전에)
            enhanced = apply_v13_enhancement(image_array, metal_type, lighting)
            
            # 3. 검은색 테두리 감지 (v20 강화 버전)
            mask, bbox, thickness = detect_black_border_v20(enhanced)
            
            # 4. 검은색 테두리 제거 (감지되면)
            if bbox and thickness > 10:
                enhanced = remove_black_border_v20(enhanced, mask, bbox, thickness)
                
                # 웨딩링 영역 추출
                ring_area = extract_ring_area(enhanced, bbox, thickness)
                
                # 웨딩링 영역만 추가 보정
                if ring_area:
                    rx, ry, rw, rh = ring_area
                    ring_region = enhanced[ry:ry+rh, rx:rx+rw].copy()
                    
                    # 웨딩링 추가 보정 (더 밝고 선명하게)
                    ring_enhanced = cv2.convertScaleAbs(ring_region, alpha=1.15, beta=10)
                    
                    # 샴페인골드면 더 화이트하게
                    if metal_type == 'champagne_gold':
                        ring_enhanced = cv2.convertScaleAbs(ring_enhanced, alpha=1.1, beta=15)
                    
                    enhanced[ry:ry+rh, rx:rx+rw] = ring_enhanced
            else:
                ring_area = None
            
            # 5. 2x 업스케일링
            height, width = enhanced.shape[:2]
            upscaled = cv2.resize(enhanced, (width * 2, height * 2), 
                                 interpolation=cv2.INTER_LANCZOS4)
            
            # 6. 썸네일 생성
            thumbnail = create_perfect_thumbnail(enhanced, bbox, ring_area)
            
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
                    "border_detected": bbox is not None,
                    "border_thickness": thickness if bbox else 0,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "version": "v20.0 Final"
                }
            }
            
    except Exception as e:
        # 에러 시에도 기본 처리
        try:
            # 최소한 밝게라도
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=20)
            main_pil = Image.fromarray(enhanced)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일도 생성
            thumbnail = cv2.resize(enhanced, (1000, 1300))
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "error": f"부분 오류 발생: {str(e)}"
            }
        except:
            return {"error": f"처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
