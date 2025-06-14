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
        
    def detect_black_lines(self, image):
        """정밀한 검은색 선 테두리 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 매우 어두운 픽셀만 검은색으로 간주 (threshold=15)
            _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
                
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
                return None, None
                
            # 가장 큰 유효한 컨투어 선택
            largest_contour = max(valid_contours, key=lambda x: x[1])
            contour, _, bbox = largest_contour
            
            # 검은색 선 마스크 생성
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)
            
            return mask, bbox
            
        except Exception as e:
            print(f"검은색 선 감지 에러: {e}")
            return None, None
    
    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 감지"""
        try:
            if mask is not None:
                # 마스킹 영역 내에서만 분석
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) == 0:
                    return 'white_gold'
                rgb_values = image[mask_indices[0], mask_indices[1], :]
                hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                avg_hue = np.mean(hsv_values[:, 0])
                avg_sat = np.mean(hsv_values[:, 1])
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
            
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
    
    def enhance_wedding_ring(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정 (28쌍 학습 데이터 기반)"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
            
            # 노이즈 제거
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # PIL ImageEnhance로 기본 보정
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
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                image, original_blend, 0
            )
            
            return final.astype(np.uint8)
            
        except Exception as e:
            print(f"웨딩링 보정 에러: {e}")
            return image
    
    def remove_black_lines_v14_9(self, image, mask, lighting):
        """v14.9: 28쌍 AFTER 파일 기준 배경색 직접 덮어쓰기"""
        try:
            result = image.copy()
            
            # 28쌍 AFTER 파일에서 검증된 배경색 사용
            target_bg_color = np.array(AFTER_BACKGROUND_COLORS[lighting], dtype=np.uint8)
            
            # 검은색 픽셀 위치 찾기 (boolean mask 방식)
            black_mask = mask > 0
            
            # 채널별로 배경색 할당 (NumPy 에러 완전 방지)
            result[black_mask, 0] = target_bg_color[0]  # R 채널
            result[black_mask, 1] = target_bg_color[1]  # G 채널
            result[black_mask, 2] = target_bg_color[2]  # B 채널
            
            # 가장자리만 부드럽게 블렌딩
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)
            edge_mask = (dilated_mask > 0) & (mask == 0)
            
            if np.any(edge_mask):
                # 가장자리 영역은 50% 블렌딩
                result[edge_mask] = (image[edge_mask] * 0.5 + target_bg_color * 0.5).astype(np.uint8)
            
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
    
    def create_thumbnail_v14_9(self, image, bbox, target_size=(1000, 1300)):
        """검은색 선 기준으로 정확한 1000×1300 썸네일 생성"""
        try:
            x, y, w, h = bbox
            
            # 웨딩링 영역에 적절한 마진 추가 (20%)
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.2)
            
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image.shape[1], x + w + margin_w)
            y2 = min(image.shape[0], y + h + margin_h)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            # 1000×1300 정확한 비율로 조정
            target_w, target_h = target_size
            crop_h, crop_w = cropped.shape[:2]
            
            # 비율 계산 (fit 방식)
            ratio_w = target_w / crop_w
            ratio_h = target_h / crop_h
            ratio = min(ratio_w, ratio_h)
            
            # 리사이즈
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000×1300 캔버스에 중앙 배치
            canvas = np.full((target_h, target_w, 3), 240, dtype=np.uint8)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            return canvas
            
        except Exception as e:
            print(f"썸네일 생성 에러: {e}")
            # 에러 시 기본 리사이즈
            return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v14.9 연결 성공: {input_data['prompt']}",
                "status": "ready_for_processing",
                "version": "v14.9_after_background_colors",
                "features": [
                    "28쌍 AFTER 파일 기준 배경색",
                    "검은색 선 직접 덮어쓰기",
                    "NumPy 에러 완전 해결",
                    "v13.3 웨딩링 보정",
                    "정확한 썸네일 1000×1300"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            processor = WeddingRingProcessor()
            
            # 1. 검은색 선 테두리 감지
            mask, bbox = processor.detect_black_lines(image_array)
            if mask is None:
                return {"error": "검은색 선 테두리를 찾을 수 없습니다."}
            
            # 2. 금속 타입 및 조명 감지 (웨딩링 영역 기준)
            x, y, w, h = bbox
            ring_region = image_array[y:y+h, x:x+w]
            metal_type = processor.detect_metal_type(ring_region)
            lighting = processor.detect_lighting(image_array)
            
            # 3. v13.3 웨딩링 보정 (검은색 선 내부만)
            enhanced_ring = processor.enhance_wedding_ring(ring_region, metal_type, lighting)
            
            # 4. 보정된 웨딩링을 원본에 다시 합성
            enhanced_image = image_array.copy()
            enhanced_image[y:y+h, x:x+w] = enhanced_ring
            
            # 5. v14.9 배경색 직접 덮어쓰기로 검은색 선 제거
            final_image = processor.remove_black_lines_v14_9(enhanced_image, mask, lighting)
            
            # 6. 2x 업스케일링
            upscaled = processor.upscale_image(final_image, scale=2)
            
            # 7. 1000×1300 썸네일 생성
            # bbox를 2배로 스케일링
            scaled_bbox = (bbox[0]*2, bbox[1]*2, bbox[2]*2, bbox[3]*2)
            thumbnail = processor.create_thumbnail_v14_9(upscaled, scaled_bbox)
            
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
                    "version": "v14.9",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "bbox": bbox,
                    "background_color": AFTER_BACKGROUND_COLORS[lighting],
                    "scale_factor": 2,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "method": "after_file_background_direct_replacement"
                }
            }
        
        return {"error": "image_base64 또는 prompt가 필요합니다."}
        
    except Exception as e:
        return {"error": f"v14.9 처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
