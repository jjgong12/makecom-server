import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 28쌍 학습 데이터 기반 파라미터 (절대 제거 금지)
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
            'original_blend': 0.12, 'saturation': 1.04, 'gamma': 1.00
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.05,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.15, 'saturation': 1.25, 'gamma': 1.02
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.10,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.15,
            'sharpness': 1.25, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.03,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.18, 'saturation': 1.28, 'gamma': 1.03
        }
    }
}

# 28쌍 AFTER 파일 배경색 데이터베이스 (28번 대화 성과 - 절대 제거 금지)
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [250, 248, 245],
        'medium': [242, 240, 237],
        'dark': [235, 233, 230]
    },
    'warm': {
        'light': [252, 248, 240],
        'medium': [248, 243, 235],
        'dark': [240, 235, 228]
    },
    'cool': {
        'light': [248, 250, 252],
        'medium': [240, 242, 245],
        'dark': [232, 235, 238]
    }
}

class UltimateWeddingRingProcessor:
    def __init__(self):
        print("UltimateWeddingRingProcessor v16.7 Complete 초기화 완료")
    
    def detect_black_border_advanced_v167(self, image):
        """v16.7: 완전히 새로운 강화된 검은색 테두리 감지 (기존 잘못된 방식 완전 대체)"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            print("강화된 검은색 테두리 감지 시작")
            
            # 1. 다중 threshold로 검은색 영역 포괄적 감지
            all_masks = []
            thresholds = [10, 15, 20, 25, 30, 35]
            
            for threshold in thresholds:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                all_masks.append(binary)
                print(f"Threshold {threshold}: {np.sum(binary > 0)} 픽셀")
            
            # 2. 마스크들을 조합해서 최적의 결과 찾기
            best_contour = None
            best_bbox = None
            best_score = 0
            
            for i, mask in enumerate(all_masks):
                # 형태학적 연산으로 노이즈 제거
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
                
                # 컨투어 찾기
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # 가장 큰 컨투어들 분석
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                    area = cv2.contourArea(contour)
                    
                    # 최소 면적 조건
                    if area < width * height * 0.08:
                        continue
                    
                    # 바운딩 박스
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 사각형 조건 체크
                    ratio = w / h if h > 0 else 0
                    perimeter = cv2.arcLength(contour, True)
                    
                    # 검은색 테두리 조건들
                    conditions = [
                        0.2 < ratio < 5.0,  # 비율 조건
                        w > width * 0.15,   # 최소 너비
                        h > height * 0.15,  # 최소 높이
                        w < width * 0.95,   # 최대 너비 (전체가 아님)
                        h < height * 0.95,  # 최대 높이 (전체가 아님)
                        area > width * height * 0.08,  # 최소 면적
                        perimeter > (width + height) * 0.3  # 최소 둘레
                    ]
                    
                    if all(conditions):
                        # 점수 계산 (더 사각형에 가까울수록 높은 점수)
                        rect_area = w * h
                        contour_area = cv2.contourArea(contour)
                        rectangularity = contour_area / rect_area if rect_area > 0 else 0
                        
                        # 가장자리에 위치할수록 높은 점수
                        edge_bonus = 0
                        if x < width * 0.1 or y < height * 0.1 or (x + w) > width * 0.9 or (y + h) > height * 0.9:
                            edge_bonus = 0.2
                        
                        score = rectangularity + edge_bonus + (area / (width * height))
                        
                        if score > best_score:
                            best_score = score
                            best_contour = contour
                            best_bbox = (x, y, w, h)
                            print(f"새로운 최고 후보: threshold={thresholds[i]}, score={score:.3f}, bbox={best_bbox}")
            
            if best_contour is not None and best_bbox is not None:
                # 최종 마스크 생성
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [best_contour], 255)
                
                print(f"검은색 테두리 감지 성공: {best_bbox}, 점수: {best_score:.3f}")
                return mask, best_bbox, best_contour
            
            print("검은색 테두리를 찾을 수 없음")
            return None, None, None
            
        except Exception as e:
            print(f"검은색 테두리 감지 오류: {e}")
            return None, None, None
    
    def detect_actual_line_thickness_safe(self, mask, bbox):
        """29번 대화 성과: 적응형 두께 감지 (절대 제거 금지)"""
        try:
            x, y, w, h = bbox
            
            # 4방향에서 실제 선 두께 측정
            thicknesses = []
            
            # 상단 측정
            if y > 10:
                for i in range(max(0, x), min(mask.shape[1], x + w), max(1, w // 20)):
                    for thickness in range(1, min(50, y)):
                        if y - thickness >= 0 and mask[y - thickness, i] == 0:
                            thicknesses.append(thickness)
                            break
            
            # 하단 측정  
            if y + h < mask.shape[0] - 10:
                for i in range(max(0, x), min(mask.shape[1], x + w), max(1, w // 20)):
                    for thickness in range(1, min(50, mask.shape[0] - y - h)):
                        if y + h + thickness < mask.shape[0] and mask[y + h + thickness, i] == 0:
                            thicknesses.append(thickness)
                            break
            
            # 좌측 측정
            if x > 10:
                for i in range(max(0, y), min(mask.shape[0], y + h), max(1, h // 20)):
                    for thickness in range(1, min(50, x)):
                        if x - thickness >= 0 and mask[i, x - thickness] == 0:
                            thicknesses.append(thickness)
                            break
            
            # 우측 측정
            if x + w < mask.shape[1] - 10:
                for i in range(max(0, y), min(mask.shape[0], y + h), max(1, h // 20)):
                    for thickness in range(1, min(50, mask.shape[1] - x - w)):
                        if x + w + thickness < mask.shape[1] and mask[i, x + w + thickness] == 0:
                            thicknesses.append(thickness)
                            break
            
            if thicknesses:
                # 중간값 사용 (outlier 제거)
                thicknesses = sorted(thicknesses)
                median_thickness = thicknesses[len(thicknesses) // 2]
                
                # 50% 안전 마진 추가
                safe_thickness = int(median_thickness * 1.5 + 10)
                print(f"적응형 두께 감지: 측정값 {median_thickness}px → 안전값 {safe_thickness}px")
                return safe_thickness
            
            print("두께 측정 실패 - 기본값 30px 사용")
            return 30
            
        except Exception as e:
            print(f"두께 감지 오류: {e}")
            return 30
    
    def detect_metal_type_enhanced(self, image, mask=None):
        """개선된 금속 타입 감지"""
        try:
            if mask is not None:
                # 마스킹 영역 내에서만 분석
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) > 0:
                    avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                    avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
                    avg_val = np.mean(hsv[mask_indices[0], mask_indices[1], 2])
                else:
                    return 'champagne_gold'
            else:
                # 중앙 영역에서 분석
                center_h, center_w = image.shape[:2]
                roi = image[center_h//4:3*center_h//4, center_w//4:3*center_w//4]
                hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
                avg_val = np.mean(hsv[:, :, 2])
            
            # 개선된 금속 타입 분류
            if avg_val > 180:  # 매우 밝은 경우
                if avg_sat < 25:
                    return 'white_gold'
                elif avg_hue < 15 or avg_hue > 165:
                    return 'rose_gold'
            
            # 일반적인 분류
            if avg_hue < 15 or avg_hue > 165:
                if avg_sat > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_hue <= 35:
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'champagne_gold'  # 기본값
                
        except Exception as e:
            print(f"금속 감지 오류: {e}")
            return 'champagne_gold'
    
    def detect_lighting_enhanced(self, image):
        """개선된 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # A, B 채널 분석
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])
            
            # RGB 채널 분석
            r_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            b_channel_mean = np.mean(image[:, :, 2])
            
            # 종합 판단
            if b_mean < 125 and r_mean > g_mean and r_mean > b_channel_mean:
                return 'warm'
            elif b_mean > 135 and b_channel_mean > r_mean:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            print(f"조명 감지 오류: {e}")
            return 'natural'
    
    def apply_v13_3_complete_enhancement(self, image, metal_type, lighting):
        """v13.3 완전한 10단계 웨딩링 보정 (절대 제거 금지)"""
        try:
            params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                     WEDDING_RING_PARAMS['champagne_gold']['natural'])
            
            print(f"v13.3 보정 시작: {metal_type}/{lighting}")
            
            # 0. 원본 보존
            original_image = image.copy()
            
            # 1. 노이즈 제거 (bilateralFilter)
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # PIL 이미지로 변환
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
            
            # 5. 채도 조정
            if 'saturation' in params:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(params['saturation'])
            
            # numpy 배열로 변환
            enhanced_array = np.array(enhanced)
            
            # 6. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 7. LAB 색공간에서 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 8. CLAHE 적용 (명료도)
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9. 감마 보정
            if 'gamma' in params:
                gamma = params['gamma']
                enhanced_array = np.power(enhanced_array / 255.0, gamma) * 255.0
                enhanced_array = enhanced_array.astype(np.uint8)
            
            # 10. 원본과 블렌딩 (자연스러움 보장)
            original_blend = params['original_blend']
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                original_image, original_blend, 0
            )
            
            print(f"v13.3 완전 보정 완료: {metal_type}/{lighting}")
            return final.astype(np.uint8)
            
        except Exception as e:
            print(f"v13.3 보정 오류: {e}")
            return image
    
    def extract_and_enhance_ring_safe(self, image, mask, bbox, metal_type, lighting):
        """25번 대화 성과: 웨딩링 확대 보정 시스템 (절대 제거 금지)"""
        try:
            x, y, w, h = bbox
            
            # 안전 마진으로 웨딩링 영역 추출
            margin_w = max(10, w // 10)
            margin_h = max(10, h // 10)
            
            safe_x = max(0, x - margin_w)
            safe_y = max(0, y - margin_h)
            safe_w = min(image.shape[1] - safe_x, w + 2 * margin_w)
            safe_h = min(image.shape[0] - safe_y, h + 2 * margin_h)
            
            # 웨딩링 영역 추출
            ring_region = image[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w].copy()
            
            # 확대 (2배)
            enlarged = cv2.resize(ring_region, (safe_w*2, safe_h*2), interpolation=cv2.INTER_LANCZOS4)
            
            # 강화된 보정 적용
            enhanced_params = WEDDING_RING_PARAMS[metal_type][lighting].copy()
            enhanced_params['brightness'] *= 1.1
            enhanced_params['contrast'] *= 1.05
            enhanced_params['sharpness'] *= 1.1
            
            # PIL로 보정
            pil_enlarged = Image.fromarray(enlarged)
            
            enhancer = ImageEnhance.Brightness(pil_enlarged)
            enhanced = enhancer.enhance(enhanced_params['brightness'])
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(enhanced_params['contrast'])
            
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(enhanced_params['sharpness'])
            
            enhanced_ring = np.array(enhanced)
            
            # 원래 크기로 축소
            final_ring = cv2.resize(enhanced_ring, (safe_w, safe_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 원본 이미지에 블렌딩
            result = image.copy()
            
            # 부드러운 블렌딩을 위한 마스크
            blend_mask = np.ones((safe_h, safe_w), dtype=np.float32)
            border_size = min(20, min(safe_w, safe_h) // 10)
            blend_mask[:border_size, :] *= np.linspace(0, 1, border_size).reshape(-1, 1)
            blend_mask[-border_size:, :] *= np.linspace(1, 0, border_size).reshape(-1, 1)
            blend_mask[:, :border_size] *= np.linspace(0, 1, border_size)
            blend_mask[:, -border_size:] *= np.linspace(1, 0, border_size)
            
            for c in range(3):
                result[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w, c] = (
                    final_ring[:, :, c] * blend_mask + 
                    image[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w, c] * (1 - blend_mask)
                )
            
            print("웨딩링 확대 보정 완료")
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"웨딩링 확대 보정 오류: {e}")
            return image
    
    def remove_black_border_with_after_background(self, image, mask, bbox, contour, border_thickness, lighting):
        """27,28번 대화 성과: 고급 inpainting + AFTER 배경색 (절대 제거 금지)"""
        try:
            x, y, w, h = bbox
            
            # 28쌍 AFTER 배경색 선택
            bg_colors = AFTER_BACKGROUND_COLORS[lighting]
            
            # 현재 배경 밝기 분석
            background_brightness = np.mean(image[mask == 0])
            
            if background_brightness > 200:
                target_bg = bg_colors['light']
            elif background_brightness > 150:
                target_bg = bg_colors['medium']
            else:
                target_bg = bg_colors['dark']
            
            print(f"AFTER 배경색 적용: {lighting} -> {target_bg}")
            
            # 웨딩링 보호 영역 설정
            inner_margin = max(border_thickness, 20)
            inner_x = max(x, x + inner_margin)
            inner_y = max(y, y + inner_margin)
            inner_w = max(1, w - 2 * inner_margin)
            inner_h = max(1, h - 2 * inner_margin)
            
            # 안전 체크
            if inner_x + inner_w > image.shape[1]:
                inner_w = image.shape[1] - inner_x
            if inner_y + inner_h > image.shape[0]:
                inner_h = image.shape[0] - inner_y
            
            if inner_w <= 0 or inner_h <= 0:
                print("웨딩링 보호 영역이 너무 작음 - 안전 처리")
                inner_x = x + w // 4
                inner_y = y + h // 4
                inner_w = w // 2
                inner_h = h // 2
            
            # 제거할 영역 마스크 생성 (웨딩링 제외)
            removal_mask = mask.copy()
            removal_mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = 0
            
            # 고급 inpainting (TELEA + NS 조합)
            inpainted_telea = cv2.inpaint(image, removal_mask, 5, cv2.INPAINT_TELEA)
            inpainted_ns = cv2.inpaint(image, removal_mask, 5, cv2.INPAINT_NS)
            
            # 두 결과 블렌딩
            inpainted = cv2.addWeighted(inpainted_telea, 0.6, inpainted_ns, 0.4, 0)
            
            # AFTER 배경색으로 보정
            bg_adjustment = np.array(target_bg) - np.mean(inpainted[removal_mask > 0], axis=0)
            bg_regions = removal_mask > 0
            
            for c in range(3):
                inpainted[bg_regions, c] = np.clip(
                    inpainted[bg_regions, c] + bg_adjustment[c] * 0.3, 0, 255
                )
            
            # 웨딩링 영역 원본 복원 (절대 보호)
            inpainted[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = \
                image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
            
            # 31×31 가우시안 블렌딩 (25번 대화 성과)
            smooth_mask = mask.astype(np.float32) / 255.0
            smooth_mask = cv2.GaussianBlur(smooth_mask, (31, 31), 10)
            
            # 부드러운 블렌딩
            result = image.copy().astype(np.float32)
            inpainted_float = inpainted.astype(np.float32)
            
            for c in range(3):
                result[:, :, c] = (
                    image[:, :, c] * (1 - smooth_mask) + 
                    inpainted_float[:, :, c] * smooth_mask
                )
            
            print("고급 inpainting + AFTER 배경색 적용 완료")
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"고급 inpainting 오류: {e}")
            return image
    
    def upscale_2x_quality(self, image):
        """고품질 2x 업스케일링"""
        try:
            height, width = image.shape[:2]
            new_width = int(width * 2)
            new_height = int(height * 2)
            
            # LANCZOS4로 고품질 업스케일링
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 추가 선명화
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            # 원본과 블렌딩 (과도한 선명화 방지)
            result = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
            
            print("2x 고품질 업스케일링 완료")
            return result
            
        except Exception as e:
            print(f"업스케일링 오류: {e}")
            return image
    
    def create_thumbnail_safe(self, image, bbox=None):
        """완벽한 1000x1300 썸네일 생성 (웨딩링 화면 가득, 여백 최소화)"""
        try:
            height, width = image.shape[:2]
            
            if bbox is not None:
                # 검은색 테두리 기준 크롭 (2배 스케일링 적용)
                x, y, w, h = bbox
                x, y, w, h = x*2, y*2, w*2, h*2  # 업스케일링 보정
                
                # 웨딩링이 더 크게 보이도록 마진 최소화
                margin_w = max(20, int(w * 0.05))  # 5% 마진
                margin_h = max(20, int(h * 0.05))  # 5% 마진
                
                crop_x = max(0, x - margin_w)
                crop_y = max(0, y - margin_h)
                crop_w = min(width - crop_x, w + 2*margin_w)
                crop_h = min(height - crop_y, h + 2*margin_h)
                
                cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                print(f"테두리 기준 크롭: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
                
            else:
                # 중앙 영역에서 밝은 부분(웨딩링) 찾기
                center_x, center_y = width // 2, height // 2
                search_w, search_h = min(width//2, 800), min(height//2, 800)
                
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                center_region = gray[center_y-search_h//2:center_y+search_h//2,
                                   center_x-search_w//2:center_x+search_w//2]
                
                # 밝은 영역 감지 (웨딩링은 일반적으로 밝음)
                threshold = np.mean(center_region) + np.std(center_region) * 0.5
                _, binary = cv2.threshold(center_region, threshold, 255, cv2.THRESH_BINARY)
                
                # 가장 큰 밝은 영역 찾기
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    rx, ry, rw, rh = cv2.boundingRect(largest_contour)
                    
                    # 실제 좌표로 변환
                    actual_x = center_x - search_w//2 + rx
                    actual_y = center_y - search_h//2 + ry
                    
                    # 마진 추가
                    margin = max(50, max(rw, rh) // 4)
                    crop_x = max(0, actual_x - margin)
                    crop_y = max(0, actual_y - margin)
                    crop_w = min(width - crop_x, rw + 2*margin)
                    crop_h = min(height - crop_y, rh + 2*margin)
                    
                    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    print(f"자동 웨딩링 감지 크롭: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
                
                else:
                    # fallback: 중앙 영역 크롭
                    crop_w, crop_h = width//2, height//2
                    crop_x, crop_y = width//4, height//4
                    cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    print("중앙 영역 크롭 적용")
            
            # 1000x1300 비율로 조정
            target_w, target_h = 1000, 1300
            crop_h, crop_w = cropped.shape[:2]
            
            # 웨딩링이 화면 가득 차도록 비율 계산
            ratio_w = target_w / crop_w
            ratio_h = target_h / crop_h
            ratio = max(ratio_w, ratio_h) * 0.92  # 웨딩링이 화면의 92% 차지
            
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            # 크기 제한
            if new_w > target_w * 1.2:
                ratio = target_w * 1.2 / crop_w
                new_w = int(crop_w * ratio)
                new_h = int(crop_h * ratio)
            
            if new_h > target_h * 1.2:
                ratio = target_h * 1.2 / crop_h
                new_w = int(crop_w * ratio)
                new_h = int(crop_h * ratio)
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000x1300 캔버스에 배치
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            
            # 위쪽에 배치 (여백 최소화)
            start_y = max(0, (target_h - new_h) // 6)  # 위쪽 1/6 지점
            start_x = max(0, (target_w - new_w) // 2)   # 가로 중앙
            
            # 캔버스 범위 확인
            end_y = min(target_h, start_y + new_h)
            end_x = min(target_w, start_x + new_w)
            actual_h = end_y - start_y
            actual_w = end_x - start_x
            
            # 배치
            canvas[start_y:end_y, start_x:end_x] = resized[:actual_h, :actual_w]
            
            print(f"완벽한 썸네일 생성 완료: 웨딩링 크기 {new_w}x{new_h}, 위치 ({start_x}, {start_y})")
            return canvas
            
        except Exception as e:
            print(f"썸네일 생성 오류: {e}")
            # 안전한 fallback
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod 메인 핸들러 v16.7 Complete"""
    try:
        input_data = event["input"]
        
        # 테스트 모드
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"v16.7 Complete 연결 성공: {input_data['prompt']}",
                "status": "ready_for_processing",
                "version": "v16.7_complete",
                "features": [
                    "강화된 검은색 테두리 감지",
                    "29번 대화 적응형 두께 감지",
                    "25번 대화 웨딩링 확대 보정", 
                    "27,28번 대화 고급 inpainting + AFTER 배경색",
                    "v13.3 완전한 10단계 보정",
                    "완벽한 썸네일 생성"
                ]
            }
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64가 필요합니다"}
        
        print("v16.7 Complete 이미지 처리 시작")
        
        # Base64 디코딩
        image_data = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        print(f"이미지 로딩 완료: {image_array.shape}")
        
        # 프로세서 초기화
        processor = UltimateWeddingRingProcessor()
        
        # 1. 금속 및 조명 감지 (개선된 버전)
        metal_type = processor.detect_metal_type_enhanced(image_array)
        lighting = processor.detect_lighting_enhanced(image_array)
        print(f"감지 완료 - 금속: {metal_type}, 조명: {lighting}")
        
        # 2. v13.3 완전한 10단계 보정 (무조건 실행)
        enhanced_image = processor.apply_v13_3_complete_enhancement(image_array, metal_type, lighting)
        print("v13.3 완전 보정 완료")
        
        # 3. 강화된 검은색 테두리 감지
        mask, bbox, contour = processor.detect_black_border_advanced_v167(enhanced_image)
        
        if mask is not None and bbox is not None:
            print("검은색 테두리 감지됨 - 고급 처리 시작")
            
            # 29번 대화: 적응형 두께 감지
            border_thickness = processor.detect_actual_line_thickness_safe(mask, bbox)
            
            # 25번 대화: 웨딩링 확대 보정
            enhanced_image = processor.extract_and_enhance_ring_safe(
                enhanced_image, mask, bbox, metal_type, lighting)
            
            # 27,28번 대화: 고급 inpainting + AFTER 배경색
            enhanced_image = processor.remove_black_border_with_after_background(
                enhanced_image, mask, bbox, contour, border_thickness, lighting)
            
            print("검은색 테두리 완전 처리 완료")
        else:
            print("검은색 테두리 없음 - v13.3 보정만 적용")
            bbox = None
        
        # 4. 고품질 2x 업스케일링
        upscaled_image = processor.upscale_2x_quality(enhanced_image)
        print("고품질 업스케일링 완료")
        
        # 5. 완벽한 썸네일 생성 (웨딩링 화면 가득)
        thumbnail = processor.create_thumbnail_safe(upscaled_image, bbox)
        print("완벽한 썸네일 생성 완료")
        
        # 6. 결과 인코딩
        # 메인 이미지
        main_pil = Image.fromarray(upscaled_image)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        print("인코딩 완료")
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "processing_info": {
                "metal_type": metal_type,
                "lighting": lighting,
                "border_detected": mask is not None,
                "bbox": bbox,
                "version": "v16.7_complete_with_all_features",
                "applied_features": [
                    "v13.3_complete_enhancement",
                    "advanced_border_detection" if mask is not None else "no_border_detected",
                    "adaptive_thickness_detection" if mask is not None else None,
                    "ring_extraction_enhancement" if mask is not None else None,
                    "advanced_inpainting_with_after_background" if mask is not None else None,
                    "quality_2x_upscaling",
                    "perfect_thumbnail_generation"
                ]
            }
        }
        
    except Exception as e:
        print(f"v16.7 Complete 처리 중 오류: {e}")
        return {"error": f"v16.7 Complete 오류: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
