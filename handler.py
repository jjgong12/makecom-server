import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 파라미터 (28쌍 학습 데이터 기반) - 완전한 12가지 세트
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12
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
            'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.05,
            'white_overlay': 0.04,
            'sharpness': 1.10,
            'color_temp_a': 0,
            'color_temp_b': -1,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.25,
            'contrast': 1.15,
            'white_overlay': 0.08,
            'sharpness': 1.25,
            'color_temp_a': 4,
            'color_temp_b': 3,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17,
            'contrast': 1.11,
            'white_overlay': 0.08,
            'sharpness': 1.16,
            'color_temp_a': -1,
            'color_temp_b': -1,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.13,
            'contrast': 1.08,
            'white_overlay': 0.10,
            'sharpness': 1.13,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.22,
            'contrast': 1.15,
            'white_overlay': 0.06,
            'sharpness': 1.20,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.12
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
            'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.06,
            'white_overlay': 0.03,
            'sharpness': 1.11,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.12,
            'white_overlay': 0.07,
            'sharpness': 1.18,
            'color_temp_a': 5,
            'color_temp_b': 4,
            'original_blend': 0.18
        }
    }
}

# 28쌍 AFTER 파일에서 추출한 검증된 배경색
AFTER_BACKGROUND_COLORS = {
    'natural': [200, 195, 190],
    'warm': [205, 198, 185], 
    'cool': [195, 198, 200]
}

class WeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        
    def detect_black_line_edges_only(self, image):
        """검은색 선 테두리만 감지 (안쪽은 제외)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 매우 어두운 픽셀만 검은색으로 간주 (threshold=15)
            _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, None
                
            # 가장 큰 사각형 모양 컨투어 찾기
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 최소 크기 조건
                    # 사각형 근사화
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # 사각형 형태
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.5 < aspect_ratio < 2.0:  # 합리적인 종횡비
                            valid_contours.append((contour, area, (x, y, w, h)))
            
            if not valid_contours:
                return None, None, None
                
            # 가장 큰 유효한 컨투어 선택
            largest_contour = max(valid_contours, key=lambda x: x[1])
            contour, _, bbox = largest_contour
            
            # 검은색 선 테두리만 마스크 생성 (안쪽 제외)
            line_mask = np.zeros_like(gray)
            cv2.drawContours(line_mask, [contour], -1, 255, thickness=3)  # 선만 그리기
            
            # 안쪽 영역 마스크 (웨딩링 감지용)
            inner_mask = np.zeros_like(gray)
            cv2.fillPoly(inner_mask, [contour], 255)
            
            return line_mask, inner_mask, bbox
            
        except Exception as e:
            print(f"검은색 선 감지 에러: {e}")
            return None, None, None
    
    def detect_wedding_ring_in_area(self, image, inner_mask):
        """검은색 선 안에서 웨딩링만 정확히 감지"""
        try:
            # 안쪽 영역에서만 웨딩링 찾기
            masked_image = cv2.bitwise_and(image, image, mask=inner_mask)
            
            # HSV로 변환
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
            
            # 금속 재질 감지 (밝고 채도 있는 영역)
            # 웨딩링은 일반적으로 밝고 반사되는 특성
            lower_metal = np.array([0, 20, 80])      # 낮은 채도, 중간 밝기
            upper_metal = np.array([180, 255, 255]) # 모든 색상 허용
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
            
            # 안쪽 영역으로 제한
            metal_mask = cv2.bitwise_and(metal_mask, inner_mask)
            
            # 형태학적 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
            metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
            
            # 가장 큰 연결된 영역만 선택 (웨딩링)
            contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # 최소 크기 확인
                    ring_mask = np.zeros_like(metal_mask)
                    cv2.fillPoly(ring_mask, [largest_contour], 255)
                    
                    # 웨딩링 바운딩 박스
                    ring_x, ring_y, ring_w, ring_h = cv2.boundingRect(largest_contour)
                    ring_bbox = (ring_x, ring_y, ring_w, ring_h)
                    
                    return ring_mask, ring_bbox
            
            return None, None
            
        except Exception as e:
            print(f"웨딩링 감지 에러: {e}")
            return None, None
    
    def detect_metal_type(self, image, ring_mask):
        """웨딩링 영역에서만 금속 타입 감지"""
        try:
            # 웨딩링 영역의 픽셀만 추출
            mask_indices = np.where(ring_mask > 0)
            if len(mask_indices[0]) == 0:
                return 'white_gold'
            
            rgb_values = image[mask_indices[0], mask_indices[1], :]
            hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            avg_hue = np.mean(hsv_values[:, 0])
            avg_sat = np.mean(hsv_values[:, 1])
            
            # 금속 타입 분류
            if avg_hue < 15 or avg_hue > 165:  # 빨간색 계열
                if avg_sat > 50:
                    return 'rose_gold'
                else:
                    return 'white_gold'
            elif 15 <= avg_hue <= 35:  # 황색 계열
                if avg_sat > 80:
                    return 'yellow_gold'
                else:
                    return 'champagne_gold'
            else:
                return 'white_gold'  # 기본값
                
        except Exception as e:
            print(f"금속 감지 에러: {e}")
            return 'white_gold'
    
    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_channel = lab[:, :, 2]
            b_mean = np.mean(b_channel)
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            print(f"조명 감지 에러: {e}")
            return 'natural'
    
    def enhance_wedding_ring_area(self, image, ring_mask, metal_type, lighting):
        """웨딩링 영역만 v13.3 보정"""
        try:
            if ring_mask is None or np.sum(ring_mask) == 0:
                return image
                
            params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
            
            result = image.copy()
            
            # 웨딩링 영역만 추출
            ring_indices = np.where(ring_mask > 0)
            if len(ring_indices[0]) == 0:
                return result
            
            # 웨딩링 영역의 바운딩 박스
            y_min, y_max = ring_indices[0].min(), ring_indices[0].max()
            x_min, x_max = ring_indices[1].min(), ring_indices[1].max()
            
            # 웨딩링 영역 크롭
            ring_region = image[y_min:y_max+1, x_min:x_max+1]
            ring_mask_crop = ring_mask[y_min:y_max+1, x_min:x_max+1]
            
            # 노이즈 제거
            denoised = cv2.bilateralFilter(ring_region, 9, 75, 75)
            
            # PIL ImageEnhance로 보정
            pil_image = Image.fromarray(denoised)
            
            # 1. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 2. 대비 조정
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 3. 선명도 조정
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # 4. 하얀색 오버레이 적용
            enhanced_array = np.array(enhanced)
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 5. LAB 색공간에서 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 6. CLAHE 명료도 개선
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 7. 감마 보정
            gamma = 1.02
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_array = cv2.LUT(enhanced_array, table)
            
            # 8. 원본과 블렌딩 (자연스러움 보장)
            original_blend = params['original_blend']
            final_region = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                ring_region, original_blend, 0
            )
            
            # 웨딩링 마스크 적용해서 원본에 합성
            mask_3d = np.stack([ring_mask_crop] * 3, axis=2) / 255.0
            result[y_min:y_max+1, x_min:x_max+1] = (
                final_region * mask_3d + 
                ring_region * (1 - mask_3d)
            ).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"웨딩링 보정 에러: {e}")
            return image
    
    def remove_black_lines_only(self, image, line_mask, lighting):
        """검은색 선만 제거하고 28쌍 AFTER 배경색으로 교체"""
        try:
            if line_mask is None or np.sum(line_mask) == 0:
                return image
                
            result = image.copy()
            
            # 28쌍 AFTER 파일에서 검증된 배경색 사용
            target_bg_color = np.array(AFTER_BACKGROUND_COLORS[lighting], dtype=np.uint8)
            
            # 검은색 선 부분만 배경색으로 교체
            line_pixels = line_mask > 0
            
            # 채널별로 배경색 적용 (NumPy 에러 방지)
            result[line_pixels, 0] = target_bg_color[0]  # R 채널
            result[line_pixels, 1] = target_bg_color[1]  # G 채널
            result[line_pixels, 2] = target_bg_color[2]  # B 채널
            
            return result
            
        except Exception as e:
            print(f"검은색 선 제거 에러: {e}")
            return image
    
    def upscale_image(self, image, scale=2):
        """LANCZOS 보간법으로 안전한 업스케일링"""
        try:
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return upscaled
        except Exception as e:
            print(f"업스케일링 에러: {e}")
            return image
    
    def create_thumbnail_ring_centered(self, image, ring_bbox, target_size=(1000, 1300)):
        """웨딩링 중심으로 정확한 1000×1300 썸네일 생성"""
        try:
            if ring_bbox is None:
                # 웨딩링을 찾지 못한 경우 중앙 크롭
                h, w = image.shape[:2]
                center_x, center_y = w // 2, h // 2
                ring_bbox = (center_x - 100, center_y - 100, 200, 200)
            
            ring_x, ring_y, ring_w, ring_h = ring_bbox
            
            # 웨딩링 중심점 계산
            ring_center_x = ring_x + ring_w // 2
            ring_center_y = ring_y + ring_h // 2
            
            # 1000x1300 비율에 맞는 크롭 영역 계산
            target_w, target_h = target_size
            
            # 웨딩링 크기 기준으로 크롭 크기 결정
            ring_size = max(ring_w, ring_h)
            crop_size = max(ring_size * 3, 400)  # 최소 400px 보장
            
            # 1000:1300 비율 유지
            crop_w = int(crop_size * target_w / target_h)
            crop_h = crop_size
            
            # 웨딩링 중심으로 크롭 영역 설정
            x1 = max(0, ring_center_x - crop_w // 2)
            y1 = max(0, ring_center_y - crop_h // 2)
            x2 = min(image.shape[1], x1 + crop_w)
            y2 = min(image.shape[0], y1 + crop_h)
            
            # 경계 처리
            if x2 - x1 < crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = max(0, y2 - crop_h)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            # 1000×1300으로 정확히 리사이즈
            thumbnail = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return thumbnail
            
        except Exception as e:
            print(f"썸네일 생성 에러: {e}")
            # 에러 시 중앙 크롭
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            crop_h, crop_w = min(h, 800), min(w, 600)
            y1 = center_y - crop_h // 2
            x1 = center_x - crop_w // 2
            cropped = image[y1:y1+crop_h, x1:x1+crop_w]
            return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.11 연결 성공: {input_data['prompt']}",
                "status": "ready_for_processing",
                "version": "v14.11_lines_only_removal",
                "features": [
                    "검은색 선 테두리만 정확히 제거",
                    "웨딩링 영역 v13.3 보정",
                    "원본 배경 완전 보존",
                    "웨딩링 중심 정확한 썸네일",
                    "28쌍 AFTER 배경색 적용"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            processor = WeddingRingProcessor()
            
            # 1. 검은색 선 테두리만 감지 (안쪽 제외)
            line_mask, inner_mask, bbox = processor.detect_black_line_edges_only(image_array)
            if line_mask is None:
                return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
            
            # 2. 검은색 선 안에서 웨딩링만 감지
            ring_mask, ring_bbox = processor.detect_wedding_ring_in_area(image_array, inner_mask)
            
            # 3. 조명 감지
            lighting = processor.detect_lighting(image_array)
            
            # 4. 웨딩링이 감지된 경우에만 보정
            enhanced_image = image_array.copy()
            if ring_mask is not None:
                # 금속 타입 감지
                metal_type = processor.detect_metal_type(image_array, ring_mask)
                
                # 웨딩링 영역 v13.3 보정
                enhanced_image = processor.enhance_wedding_ring_area(
                    image_array, ring_mask, metal_type, lighting
                )
            else:
                metal_type = 'white_gold'  # 기본값
            
            # 5. 검은색 선만 제거 (나머지는 원본 유지)
            final_image = processor.remove_black_lines_only(enhanced_image, line_mask, lighting)
            
            # 6. 2x 업스케일링
            upscaled = processor.upscale_image(final_image, scale=2)
            
            # 7. 웨딩링 중심으로 정확한 1000×1300 썸네일 생성
            if ring_bbox is not None:
                # ring_bbox를 2배로 스케일링
                scaled_ring_bbox = (ring_bbox[0]*2, ring_bbox[1]*2, ring_bbox[2]*2, ring_bbox[3]*2)
            else:
                scaled_ring_bbox = None
            thumbnail = processor.create_thumbnail_ring_centered(upscaled, scaled_ring_bbox)
            
            # 8. 결과 인코딩
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
                    "version": "v14.11",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "bbox": bbox,
                    "ring_bbox": ring_bbox,
                    "ring_detected": ring_mask is not None,
                    "background_color": AFTER_BACKGROUND_COLORS[lighting],
                    "scale_factor": 2,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "method": "line_edges_only_removal"
                }
            }
        
        return {"error": "image_base64 또는 prompt가 필요합니다."}
        
    except Exception as e:
        return {"error": f"v14.11 처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
