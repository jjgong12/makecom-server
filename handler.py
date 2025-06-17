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
        'light': [235, 232, 228],   # 너무 밝지 않은 배경
        'medium': [228, 225, 221],
        'default': [235, 232, 228]
    },
    'warm': {
        'light': [238, 233, 225],
        'medium': [232, 227, 219],
        'default': [238, 233, 225]
    },
    'cool': {
        'light': [232, 235, 238],
        'medium': [225, 228, 232],
        'default': [232, 235, 238]
    }
}

def detect_border_ultra_adaptive(image):
    """대화 29번: 적응형 검은색 테두리 감지 (100픽셀 두께 대응)"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 다중 threshold로 검은색 감지 (5-60)
    combined_mask = np.zeros_like(gray)
    for thresh in range(5, 61, 5):
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    
    # 가장자리 200픽셀까지 확인 (100픽셀 두께 + 여유)
    edge_mask = np.zeros_like(gray)
    edge_width = 200
    edge_mask[:edge_width, :] = 255  # 상단
    edge_mask[-edge_width:, :] = 255  # 하단
    edge_mask[:, :edge_width] = 255  # 좌측
    edge_mask[:, -edge_width:] = 255  # 우측
    
    # 가장자리에서만 검은색 찾기
    border_mask = cv2.bitwise_and(combined_mask, edge_mask)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, kernel)
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_OPEN, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    # 가장 큰 컨투어 선택
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    # 실제 두께 측정 (대화 29번 방식)
    thickness = detect_actual_line_thickness(border_mask, (x, y, bw, bh))
    
    return (x, y, bw, bh), thickness, largest_contour

def detect_actual_line_thickness(mask, bbox):
    """대화 29번: 실제 선 두께 측정"""
    x, y, w, h = bbox
    thicknesses = []
    
    # 상단 두께
    for i in range(min(150, h//4)):
        if np.sum(mask[y+i, x:x+w]) < w * 255 * 0.5:
            thicknesses.append(i)
            break
    
    # 하단 두께
    for i in range(min(150, h//4)):
        if np.sum(mask[y+h-i-1, x:x+w]) < w * 255 * 0.5:
            thicknesses.append(i)
            break
    
    # 좌측 두께
    for i in range(min(150, w//4)):
        if np.sum(mask[y:y+h, x+i]) < h * 255 * 0.5:
            thicknesses.append(i)
            break
    
    # 우측 두께
    for i in range(min(150, w//4)):
        if np.sum(mask[y:y+h, x+w-i-1]) < h * 255 * 0.5:
            thicknesses.append(i)
            break
    
    if thicknesses:
        # 최대값 사용 + 50% 안전 마진
        return int(max(thicknesses) * 1.5) + 20
    else:
        return 50  # 기본값

def detect_metal_type_advanced(image, mask=None):
    """금속 타입 감지 (샴페인골드 우선)"""
    if mask is not None:
        # 마스크 내부에서만 분석
        masked = cv2.bitwise_and(image, image, mask=mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
        
        # 마스크 영역만 선택
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0:
            h_values = hsv[mask_indices[0], mask_indices[1], 0]
            s_values = hsv[mask_indices[0], mask_indices[1], 1]
            v_values = hsv[mask_indices[0], mask_indices[1], 2]
            
            h_mean = np.mean(h_values)
            s_mean = np.mean(s_values)
            v_mean = np.mean(v_values)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1])
        v_mean = np.mean(hsv[:,:,2])
    
    # 샴페인골드 우선 감지
    if 15 <= h_mean <= 35 and s_mean < 60 and v_mean > 150:
        return 'champagne_gold'
    elif s_mean < 30 and v_mean > 180:
        return 'white_gold'
    elif h_mean < 15 and s_mean > 50:
        return 'rose_gold'
    elif h_mean > 20 and s_mean > 60:
        return 'yellow_gold'
    else:
        # 애매하면 샴페인골드로
        return 'champagne_gold'

def detect_lighting_condition(image):
    """조명 환경 감지"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_mean = np.mean(lab[:,:,0])
    a_mean = np.mean(lab[:,:,1])
    b_mean = np.mean(lab[:,:,2])
    
    # B 채널로 따뜻함/차가움 판단
    if b_mean < 125:
        return 'warm'
    elif b_mean > 135:
        return 'cool'
    else:
        return 'natural'

def apply_v13_complete_enhancement(image, params):
    """v13.3 완전한 10단계 보정 프로세스"""
    original = image.copy()
    
    # 1. 노이즈 제거 (대화 16번)
    enhanced = cv2.bilateralFilter(image, 9, 75, 75)
    
    # PIL 변환
    pil_img = Image.fromarray(enhanced)
    
    # 2. 밝기 조정
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(params['brightness'])
    
    # 3. 대비 조정
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(params['contrast'])
    
    # 4. 선명도 조정
    sharpness_enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = sharpness_enhancer.enhance(params['sharpness'])
    
    # numpy 변환
    enhanced = np.array(pil_img)
    
    # 5. 채도 조정 (HSV)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * params.get('saturation', 1.0)
    hsv[:,:,1][hsv[:,:,1] > 255] = 255
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 6. 하얀색 오버레이 (대화 9번: "하얀색 살짝 입힌 느낌")
    white_overlay = np.full_like(enhanced, 255)
    enhanced = cv2.addWeighted(enhanced, 1 - params['white_overlay'], 
                              white_overlay, params['white_overlay'], 0)
    
    # 7. 색온도 조정 (LAB)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:,:,1] = np.clip(lab[:,:,1] + params['color_temp_a'], 0, 255)
    lab[:,:,2] = np.clip(lab[:,:,2] + params['color_temp_b'], 0, 255)
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # 8. CLAHE (명료도)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 9. 감마 보정
    gamma = params.get('gamma', 1.0)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
    enhanced = cv2.LUT(enhanced, table)
    
    # 10. 원본 블렌딩
    final = cv2.addWeighted(enhanced, 1 - params['original_blend'], 
                           original, params['original_blend'], 0)
    
    return final

def remove_border_with_after_background(image, bbox, thickness, lighting='natural'):
    """대화 27-28번: 검은색 선 제거 + 28쌍 AFTER 배경색"""
    x, y, w, h = bbox
    result = image.copy()
    
    # 28쌍 AFTER 배경색 가져오기
    bg_color = np.array(AFTER_BACKGROUND_COLORS[lighting]['default'])
    
    # 제거할 두께 (실제 두께 + 안전 마진)
    remove_thickness = thickness + 30
    
    # 테두리 영역을 배경색으로 교체
    # 상단
    result[:y+remove_thickness, :] = bg_color
    # 하단
    result[y+h-remove_thickness:, :] = bg_color
    # 좌측
    result[:, :x+remove_thickness] = bg_color
    # 우측
    result[:, x+w-remove_thickness:] = bg_color
    
    # 대화 31번: 31x31 가우시안 블러로 자연스러운 블렌딩
    # 블렌딩 마스크 생성
    blend_mask = np.zeros(image.shape[:2], dtype=np.float32)
    blend_thickness = 20
    
    # 가장자리 블렌딩 영역 설정
    # 상단
    if y+remove_thickness < image.shape[0]:
        blend_mask[y+remove_thickness-blend_thickness:y+remove_thickness+blend_thickness, x:x+w] = 1.0
    # 하단
    if y+h-remove_thickness > 0:
        blend_mask[y+h-remove_thickness-blend_thickness:y+h-remove_thickness+blend_thickness, x:x+w] = 1.0
    # 좌측
    if x+remove_thickness < image.shape[1]:
        blend_mask[y:y+h, x+remove_thickness-blend_thickness:x+remove_thickness+blend_thickness] = 1.0
    # 우측
    if x+w-remove_thickness > 0:
        blend_mask[y:y+h, x+w-remove_thickness-blend_thickness:x+w-remove_thickness+blend_thickness] = 1.0
    
    # 31x31 가우시안 블러
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 10)
    
    # 블렌딩 적용
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - blend_mask) + image[:,:,c] * blend_mask
    
    return result.astype(np.uint8)

def create_perfect_thumbnail(image, bbox=None):
    """대화 34번: 완벽한 1000x1300 썸네일 (웨딩링 화면 가득)"""
    h, w = image.shape[:2]
    
    if bbox:
        x, y, bw, bh = bbox
        # 웨딩링 중심
        center_x = x + bw // 2
        center_y = y + bh // 2
        
        # 크롭 크기 (웨딩링이 화면의 80% 차지하도록)
        crop_size = int(max(bw, bh) * 1.2)
    else:
        # 테두리 없으면 중앙 기준
        center_x = w // 2
        center_y = h // 2
        crop_size = min(h, w) // 2
    
    # 크롭 영역
    x1 = max(0, center_x - crop_size)
    y1 = max(0, center_y - crop_size)
    x2 = min(w, center_x + crop_size)
    y2 = min(h, center_y + crop_size)
    
    cropped = image[y1:y2, x1:x2]
    
    # 1000x1300으로 리사이즈
    thumbnail = cv2.resize(cropped, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail

def handler(event):
    """RunPod Serverless 핸들러 - 42개 대화 모든 성과 포함"""
    try:
        input_data = event.get("input", {})
        
        # 테스트
        if "test" in input_data:
            return {
                "output": {
                    "status": "ready",
                    "message": "Wedding Ring AI v22.0 Strong Complete - 42개 대화 성과 모두 포함",
                    "version": "v22.0",
                    "features": [
                        "v13.3 완전한 12가지 파라미터",
                        "10단계 보정 프로세스",
                        "적응형 두께 감지 (100픽셀)",
                        "28쌍 AFTER 배경색",
                        "샴페인골드 화이트화",
                        "31x31 가우시안 블렌딩",
                        "완벽한 1000x1300 썸네일"
                    ]
                }
            }
        
        # 이미지 처리
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            return {"output": {"error": "No image provided"}}
        
        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        original_image = image_array.copy()
        
        # 1. 검은색 테두리 감지 (대화 29번: 적응형)
        bbox, thickness, contour = detect_border_ultra_adaptive(image_array)
        
        # 2. 금속 타입 감지 (샴페인골드 우선)
        if bbox and contour is not None:
            # 테두리 내부에서만 감지
            inner_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
            cv2.drawContours(inner_mask, [contour], -1, 255, -1)
            metal_type = detect_metal_type_advanced(image_array, inner_mask)
        else:
            metal_type = detect_metal_type_advanced(image_array)
        
        # 3. 조명 환경 감지
        lighting = detect_lighting_condition(image_array)
        
        # 4. 파라미터 선택
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                         WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 5. v13.3 완전한 10단계 보정 적용
        enhanced = apply_v13_complete_enhancement(image_array, params)
        
        # 6. 검은색 테두리 제거 (있을 경우)
        if bbox and thickness > 10:
            enhanced = remove_border_with_after_background(enhanced, bbox, thickness, lighting)
            
            # 웨딩링 영역 추가 보정
            x, y, w, h = bbox
            ring_margin = thickness + 50
            ring_x = x + ring_margin
            ring_y = y + ring_margin
            ring_w = w - 2*ring_margin
            ring_h = h - 2*ring_margin
            
            if ring_w > 50 and ring_h > 50:
                # 웨딩링 영역만 추가로 밝고 선명하게
                ring_area = enhanced[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w]
                ring_area = cv2.convertScaleAbs(ring_area, alpha=1.1, beta=5)
                
                # 언샤프 마스킹으로 디테일 강화
                gaussian = cv2.GaussianBlur(ring_area, (0, 0), 2.0)
                ring_area = cv2.addWeighted(ring_area, 1.5, gaussian, -0.5, 0)
                
                enhanced[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w] = ring_area
        
        # 7. 2x 업스케일링 (LANCZOS4)
        height, width = enhanced.shape[:2]
        upscaled = cv2.resize(enhanced, (width*2, height*2), interpolation=cv2.INTER_LANCZOS4)
        
        # 8. 썸네일 생성 (대화 34번: 웨딩링 화면 가득)
        thumbnail = create_perfect_thumbnail(enhanced, bbox)
        
        # 9. JPEG 인코딩 (품질 95)
        _, main_buffer = cv2.imencode('.jpg', upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        _, thumb_buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 10. Base64 인코딩
        main_base64 = base64.b64encode(main_buffer).decode('utf-8')
        thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
        
        # output 중첩 구조 (Make.com 경로에 맞춤)
        return {
            "output": {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": bbox is not None,
                    "border_thickness": thickness if bbox else 0,
                    "v13_params_used": True,
                    "after_background_used": bbox is not None,
                    "processing_steps": [
                        "border_detection",
                        "metal_detection",
                        "lighting_analysis",
                        "v13.3_enhancement",
                        "border_removal",
                        "ring_enhancement",
                        "upscaling",
                        "thumbnail_creation"
                    ]
                }
            }
        }
        
    except Exception as e:
        # 에러 발생시에도 최소한의 처리
        try:
            # 기본 보정이라도 적용
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.2, beta=10)
            _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fallback_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "output": {
                    "enhanced_image": fallback_base64,
                    "thumbnail": fallback_base64,
                    "processing_info": {
                        "error": str(e),
                        "fallback_used": True
                    }
                }
            }
        except:
            return {
                "output": {
                    "error": f"Critical error: {str(e)}",
                    "enhanced_image": "",
                    "thumbnail": ""
                }
            }

# RunPod Serverless 시작
runpod.serverless.start({"handler": handler})
