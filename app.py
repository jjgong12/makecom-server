import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 파라미터 (28쌍 학습 데이터 기반 - 완전한 12가지 세트)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3, 'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5, 'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2, 'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1, 'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.05, 'white_overlay': 0.04,
            'sharpness': 1.12, 'color_temp_a': 1, 'color_temp_b': 0, 'original_blend': 0.22
        },
        'cool': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.08,
            'sharpness': 1.18, 'color_temp_a': 3, 'color_temp_b': 2, 'original_blend': 0.18
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,  # v15.3 화이트화
            'sharpness': 1.16, 'color_temp_a': -6, 'color_temp_b': -6, 'original_blend': 0.15  # 화이트골드 방향
        },
        'warm': {
            'brightness': 1.14, 'contrast': 1.08, 'white_overlay': 0.10,
            'sharpness': 1.14, 'color_temp_a': -4, 'color_temp_b': -4, 'original_blend': 0.17
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.14,
            'sharpness': 1.18, 'color_temp_a': -8, 'color_temp_b': -8, 'original_blend': 0.13
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2, 'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.13, 'contrast': 1.06, 'white_overlay': 0.03,
            'sharpness': 1.12, 'color_temp_a': 2, 'color_temp_b': 1, 'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.19, 'contrast': 1.12, 'white_overlay': 0.07,
            'sharpness': 1.16, 'color_temp_a': 4, 'color_temp_b': 3, 'original_blend': 0.20
        }
    }
}

def detect_black_lines_ultimate(image):
    """적응형 검은색 선 감지 (v15.3 완전 안전 버전)"""
    try:
        h, w = image.shape[:2]
        
        # 최소 크기 검증
        if h < 100 or w < 100:
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 다중 threshold 검은색 감지 (v15.3 적응형)
        masks = []
        for threshold in [20, 30, 40]:
            mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
            masks.append(mask)
        
        # 모든 마스크 결합
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 가장 적절한 직사각형 찾기
        best_rect = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 너무 작은 영역 제외
                continue
            
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # 안전한 경계 체크
            if (x < 0 or y < 0 or x + w_rect > w or y + h_rect > h):
                continue
            
            # 종횡비 체크 (정상적인 직사각형)
            if h_rect > 0 and w_rect > 0:
                aspect_ratio = w_rect / h_rect
                if 0.2 <= aspect_ratio <= 5.0:
                    score = area * (1.0 / abs(aspect_ratio - 1.0) if abs(aspect_ratio - 1.0) > 0.1 else 10.0)
                    if score > best_score:
                        best_score = score
                        best_rect = (x, y, w_rect, h_rect)
        
        if best_rect is None:
            return None, None
        
        # 선택된 영역의 마스크 생성
        x, y, w_rect, h_rect = best_rect
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (x, y), (x + w_rect, y + h_rect), 255, -1)
        
        return mask, best_rect
        
    except Exception:
        return None, None

def detect_actual_line_thickness_safe(combined_mask, bbox):
    """v15.3 적응형 선 두께 감지 (완전 안전)"""
    try:
        x, y, w, h = bbox
        img_h, img_w = combined_mask.shape
        
        # 안전한 샘플링 영역들
        thickness_samples = []
        
        # 상단 선 (안전한 범위만)
        if y >= 15 and y + 20 < img_h:
            try:
                top_region = combined_mask[max(0, y-10):min(img_h, y+20), max(0, x):min(img_w, x+w)]
                if top_region.size > 0:
                    top_thickness = np.sum(top_region > 0, axis=0)
                    valid_thickness = top_thickness[top_thickness > 0]
                    if len(valid_thickness) > 0:
                        thickness_samples.extend(valid_thickness[:50])  # 최대 50개 샘플
            except:
                pass
        
        # 하단 선
        if y + h >= 15 and y + h + 20 < img_h:
            try:
                bottom_region = combined_mask[max(0, y+h-20):min(img_h, y+h+10), max(0, x):min(img_w, x+w)]
                if bottom_region.size > 0:
                    bottom_thickness = np.sum(bottom_region > 0, axis=0)
                    valid_thickness = bottom_thickness[bottom_thickness > 0]
                    if len(valid_thickness) > 0:
                        thickness_samples.extend(valid_thickness[:50])
            except:
                pass
        
        # 좌측 선
        if x >= 15 and x + 20 < img_w:
            try:
                left_region = combined_mask[max(0, y):min(img_h, y+h), max(0, x-10):min(img_w, x+20)]
                if left_region.size > 0:
                    left_thickness = np.sum(left_region > 0, axis=1)
                    valid_thickness = left_thickness[left_thickness > 0]
                    if len(valid_thickness) > 0:
                        thickness_samples.extend(valid_thickness[:50])
            except:
                pass
        
        # 우측 선
        if x + w >= 15 and x + w + 20 < img_w:
            try:
                right_region = combined_mask[max(0, y):min(img_h, y+h), max(0, x+w-20):min(img_w, x+w+10)]
                if right_region.size > 0:
                    right_thickness = np.sum(right_region > 0, axis=1)
                    valid_thickness = right_thickness[right_thickness > 0]
                    if len(valid_thickness) > 0:
                        thickness_samples.extend(valid_thickness[:50])
            except:
                pass
        
        # 안전한 두께 계산
        if len(thickness_samples) == 0:
            return 80  # 안전한 기본값
        
        # 이상치 제거 및 중간값 계산
        filtered_samples = [t for t in thickness_samples if 30 <= t <= 180]
        if len(filtered_samples) == 0:
            return 80
        
        # 중간값으로 안정적 계산
        thickness = int(np.median(filtered_samples))
        
        # 안전한 범위로 제한
        return max(50, min(150, thickness))
        
    except Exception:
        return 80  # 모든 예외에 대해 안전한 기본값

def detect_metal_type_safe(image, mask=None):
    """안전한 금속 타입 감지"""
    try:
        if mask is not None:
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return 'white_gold'
            
            # 마스킹된 영역에서만 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
            avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
        
        # 금속 분류 로직
        if avg_hue < 15 or avg_hue > 165:
            return 'rose_gold' if avg_sat > 50 else 'white_gold'
        elif 15 <= avg_hue <= 35:
            return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
        else:
            return 'white_gold'
            
    except Exception:
        return 'white_gold'

def detect_lighting_safe(image):
    """안전한 조명 환경 감지"""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_mean = np.mean(lab[:, :, 2])
        
        if b_mean < 125:
            return 'warm'
        elif b_mean > 135:
            return 'cool'
        else:
            return 'natural'
    except Exception:
        return 'natural'

def enhance_wedding_ring_v13_3_optimized(image, metal_type, lighting):
    """v13.3 웨딩링 보정 (메모리 최적화 버전)"""
    try:
        # 파라미터 가져오기
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # PIL 변환 및 기본 보정
        pil_image = Image.fromarray(image)
        
        # 1-3. 기본 보정 (밝기, 대비, 선명도)
        enhanced = ImageEnhance.Brightness(pil_image).enhance(params['brightness'])
        enhanced = ImageEnhance.Contrast(enhanced).enhance(params['contrast'])
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(params['sharpness'])
        
        # NumPy 변환
        enhanced_array = np.array(enhanced)
        
        # 4. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
        white_strength = params['white_overlay']
        enhanced_array = enhanced_array.astype(np.float32)
        enhanced_array = enhanced_array * (1 - white_strength) + 255 * white_strength
        enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
        
        # 5. LAB 색공간 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab = lab.astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 6. 원본과 블렌딩 (자연스러움 보장)
        blend_ratio = params['original_blend']
        enhanced_array = enhanced_array.astype(np.float32)
        image_float = image.astype(np.float32)
        final = enhanced_array * (1 - blend_ratio) + image_float * blend_ratio
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # 7. 노이즈 제거 (가벼운 양방향 필터)
        final = cv2.bilateralFilter(final, 5, 50, 50)
        
        # 8. CLAHE 적용 (제한적)
        lab = cv2.cvtColor(final, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 9. 감마 보정 (미세)
        gamma_table = np.array([((i / 255.0) ** (1.0/1.03)) * 255 for i in range(256)]).astype("uint8")
        final = cv2.LUT(final, gamma_table)
        
        # 10. 하이라이트 부스팅 (제한적)
        gray = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        highlight_threshold = np.percentile(gray, 88)
        highlight_mask = (gray > highlight_threshold).astype(np.float32) * 0.06
        
        final = final.astype(np.float32)
        for c in range(3):
            final[:, :, c] = np.clip(final[:, :, c] * (1 + highlight_mask), 0, 255)
        
        return final.astype(np.uint8)
        
    except Exception:
        return image

def extract_and_enhance_ring_ultimate(image, line_mask, bbox):
    """v15.3 적응형 웨딩링 추출 및 보정"""
    try:
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # 실제 선 두께 측정 (v15.3 적응형)
        border_thickness = detect_actual_line_thickness_safe(line_mask, bbox)
        
        # 적응형 마진 계산 (v15.3 핵심)
        # 실제 측정값 기반 + 50% 안전 마진
        adaptive_margin = max(30, border_thickness + border_thickness // 2)
        
        # 최종 안전 마진 (더 보수적으로)
        safety_margin = min(adaptive_margin, min(w, h) // 4)  # 너비/높이의 1/4을 넘지 않음
        
        # 내부 영역 계산
        inner_x = max(0, x + safety_margin)
        inner_y = max(0, y + safety_margin)
        inner_x2 = min(img_w, x + w - safety_margin)
        inner_y2 = min(img_h, y + h - safety_margin)
        
        inner_w = inner_x2 - inner_x
        inner_h = inner_y2 - inner_y
        
        # 영역 크기 검증 및 자동 조정
        if inner_w <= 60 or inner_h <= 60:
            # 더 작은 마진으로 재계산
            safety_margin = max(20, border_thickness // 3)
            inner_x = max(0, x + safety_margin)
            inner_y = max(0, y + safety_margin)
            inner_x2 = min(img_w, x + w - safety_margin)
            inner_y2 = min(img_h, y + h - safety_margin)
            inner_w = inner_x2 - inner_x
            inner_h = inner_y2 - inner_y
            
            if inner_w <= 30 or inner_h <= 30:
                # 최소한의 보정으로 전환
                return image
        
        # 웨딩링 영역 추출
        ring_region = image[inner_y:inner_y2, inner_x:inner_x2].copy()
        
        # 금속 타입 및 조명 감지
        metal_type = detect_metal_type_safe(ring_region)
        lighting = detect_lighting_safe(ring_region)
        
        # v13.3 보정 적용
        enhanced_ring = enhance_wedding_ring_v13_3_optimized(ring_region, metal_type, lighting)
        
        # 결과 이미지 생성
        result = image.copy()
        result[inner_y:inner_y2, inner_x:inner_x2] = enhanced_ring
        
        return result
        
    except Exception:
        return image

def inpaint_black_lines_ultimate(image, line_mask):
    """v15.3 검은색 선 제거 (안전하고 효과적)"""
    try:
        if line_mask is None or np.sum(line_mask) == 0:
            return image
        
        # 마스크 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # 인페인팅 실행 (NS 방식으로 자연스럽게)
        inpainted = cv2.inpaint(image, cleaned_mask, 3, cv2.INPAINT_NS)
        
        # 결과 검증
        if inpainted is None or inpainted.shape != image.shape:
            return image
        
        # 가장자리 부드럽게 처리
        dilated_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)
        edge_mask = dilated_mask - cleaned_mask
        
        if np.any(edge_mask):
            # 가장자리만 가우시안 블러
            blurred = cv2.GaussianBlur(inpainted, (5, 5), 0)
            edge_indices = np.where(edge_mask > 0)
            inpainted[edge_indices] = blurred[edge_indices]
        
        return inpainted
        
    except Exception:
        return image

def create_thumbnail_ultimate(image, bbox):
    """v15.3 완벽한 썸네일 생성 (1000×1300)"""
    try:
        if bbox is None:
            # bbox 없으면 전체 이미지 리사이즈
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # 썸네일용 여유 공간 (웨딩링이 80% 차지하도록)
        target_fill_ratio = 0.8
        margin_ratio = (1.0 - target_fill_ratio) / 2.0
        
        margin_w = max(50, int(w * margin_ratio / target_fill_ratio))
        margin_h = max(50, int(h * margin_ratio / target_fill_ratio))
        
        # 크롭 영역 계산
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)
        
        # 크롭 실행
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        # 1000×1300 비율로 조정
        crop_h, crop_w = cropped.shape[:2]
        if crop_h == 0 or crop_w == 0:
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        # 비율 맞추기 (1000:1300 = 0.769)
        target_ratio = 1000.0 / 1300.0
        current_ratio = crop_w / crop_h
        
        if current_ratio > target_ratio:
            # 너무 넓음 -> 높이 기준
            new_h = crop_h
            new_w = int(crop_h * target_ratio)
        else:
            # 너무 높음 -> 너비 기준
            new_w = crop_w
            new_h = int(crop_w / target_ratio)
        
        # 중앙에서 크롭
        start_x = max(0, (crop_w - new_w) // 2)
        start_y = max(0, (crop_h - new_h) // 2)
        
        final_crop = cropped[start_y:start_y+new_h, start_x:start_x+new_w]
        
        if final_crop.size == 0:
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        # 최종 1000×1300 리사이즈
        thumbnail = cv2.resize(final_crop, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        
        return thumbnail
        
    except Exception:
        # 모든 예외에 대해 안전한 fallback
        try:
            return cv2.resize(image, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
        except:
            return image

def handler(event):
    """RunPod Serverless 메인 핸들러 - v15.3.2 최종 안정화 버전"""
    try:
        # 입력 검증
        input_data = event.get("input", {})
        
        # 연결 테스트
        if "test" in input_data:
            return {
                "success": True,
                "message": "v15.3.2 Final 연결 성공",
                "version": "v15.3.2",
                "features": [
                    "적응형 검은색 선 감지 (100픽셀 두께 대응)",
                    "v13.3 웨딩링 보정 (28쌍 데이터)",
                    "샴페인골드 화이트화",
                    "완벽한 1000×1300 썸네일",
                    "RunPod 크래시 완전 해결"
                ]
            }
        
        # 이미지 처리
        if "image_base64" not in input_data:
            return {"error": "image_base64 필드가 필요합니다"}
        
        # Base64 디코딩 (메모리 효율적)
        try:
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return {"error": f"이미지 디코딩 실패: {str(e)}"}
        
        # 1. 검은색 선 감지 (v15.3 적응형)
        line_mask, bbox = detect_black_lines_ultimate(image_array)
        
        if line_mask is None or bbox is None:
            # 검은색 선 없는 경우 기본 보정
            metal_type = detect_metal_type_safe(image_array)
            lighting = detect_lighting_safe(image_array)
            enhanced_image = enhance_wedding_ring_v13_3_optimized(image_array, metal_type, lighting)
            
            # 2x 업스케일링
            height, width = enhanced_image.shape[:2]
            upscaled = cv2.resize(enhanced_image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # 기본 썸네일
            thumbnail = create_thumbnail_ultimate(upscaled, None)
            final_image = upscaled
            
        else:
            # 2. 웨딩링 추출 및 보정 (v15.3 적응형)
            enhanced_image = extract_and_enhance_ring_ultimate(image_array, line_mask, bbox)
            
            # 3. 2x 업스케일링
            height, width = enhanced_image.shape[:2]
            upscaled = cv2.resize(enhanced_image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
            
            # 업스케일된 마스크
            upscaled_mask = cv2.resize(line_mask, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
            upscaled_mask = np.where(upscaled_mask > 127, 255, 0).astype(np.uint8)
            
            # 4. 검은색 선 제거 (v15.3 완전 제거)
            final_image = inpaint_black_lines_ultimate(upscaled, upscaled_mask)
            
            # 5. 썸네일 생성 (업스케일된 bbox 기준)
            scaled_bbox = (bbox[0] * 2, bbox[1] * 2, bbox[2] * 2, bbox[3] * 2)
            thumbnail = create_thumbnail_ultimate(final_image, scaled_bbox)
        
        # 결과 인코딩 (메모리 효율적)
        try:
            # 메인 이미지
            main_pil = Image.fromarray(final_image)
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
                    "version": "v15.3.2-final",
                    "black_lines_detected": bbox is not None,
                    "adaptive_thickness": bbox is not None,
                    "bbox": bbox,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "champagne_gold": "화이트화 적용",
                    "features": "v15.3 적응형 시스템"
                }
            }
            
        except Exception as e:
            return {"error": f"결과 인코딩 실패: {str(e)}"}
    
    except Exception as e:
        return {
            "error": f"처리 중 오류: {str(e)}",
            "version": "v15.3.2-final"
        }

# RunPod 서버리스 시작 (RunPod 호환성 최대화)
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
