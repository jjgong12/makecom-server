import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반) - 대화 16-20에서 완성
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
            'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.03, 'gamma': 1.02
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
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.15,  # 대화 29: 0.12→0.15
            'sharpness': 1.16, 'color_temp_a': -6, 'color_temp_b': -6,   # 대화 29: -4→-6 화이트화
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.13,
            'sharpness': 1.20, 'color_temp_a': -8, 'color_temp_b': -8,
            'original_blend': 0.20, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.17,
            'sharpness': 1.25, 'color_temp_a': -5, 'color_temp_b': -5,
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
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 3,
            'original_blend': 0.18, 'saturation': 1.28, 'gamma': 1.03
        }
    }
}

# 28쌍 AFTER 파일들의 배경색 (대화 28번) - 더 밝고 깔끔하게
AFTER_BACKGROUND_COLORS = {
    'natural': {'light': [250, 248, 245], 'medium': [242, 240, 237]},
    'warm': {'light': [252, 250, 245], 'medium': [245, 242, 237]},
    'cool': {'light': [248, 250, 252], 'medium': [240, 242, 245]}
}

class UltimateWeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        self.after_bg_colors = AFTER_BACKGROUND_COLORS
    
    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 자동 감지"""
        try:
            if mask is not None:
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) > 0:
                    avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                    avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
                else:
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    avg_hue = np.mean(hsv[:, :, 0])
                    avg_sat = np.mean(hsv[:, :, 1])
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
            
            # 보수적 금속 감지 - 애매하면 샴페인골드
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
                return 'champagne_gold'  # 기본값
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_channel_mean = np.mean(lab[:, :, 2])
            
            if b_channel_mean < 123:
                return 'warm'
            elif b_channel_mean > 133:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def detect_black_border_complete(self, image):
        """검은색 테두리 정확히 감지 (대화 29번 방식)"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 전체 이미지에서 검은색 영역 찾기 (threshold 높임)
            combined_mask = np.zeros_like(gray)
            
            # 다중 threshold (더 높은 값 사용)
            for threshold in [15, 25, 35, 50]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                combined_mask = cv2.bitwise_or(combined_mask, binary)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 이미지 크기의 50% 이상인 경우만 유효한 테두리로 간주
                if w > width * 0.5 and h > height * 0.5:
                    # 실제 검은색 선 두께 측정
                    thickness = self.measure_border_thickness(combined_mask, x, y, w, h)
                    return combined_mask, (x, y, w, h), thickness
            
            return None, None, 0
        except:
            return None, None, 0
    
    def measure_border_thickness(self, mask, x, y, w, h):
        """실제 검은색 선 두께 정확히 측정 (대화 29번 핵심)"""
        try:
            thicknesses = []
            
            # 상단 선 두께
            for i in range(min(200, h//2)):
                row = mask[y+i, x:x+w]
                if np.sum(row) < w * 255 * 0.8:
                    if i > 0:
                        thicknesses.append(i)
                    break
            
            # 하단 선 두께
            for i in range(min(200, h//2)):
                row = mask[y+h-1-i, x:x+w]
                if np.sum(row) < w * 255 * 0.8:
                    if i > 0:
                        thicknesses.append(i)
                    break
            
            # 좌측 선 두께
            for i in range(min(200, w//2)):
                col = mask[y:y+h, x+i]
                if np.sum(col) < h * 255 * 0.8:
                    if i > 0:
                        thicknesses.append(i)
                    break
            
            # 우측 선 두께
            for i in range(min(200, w//2)):
                col = mask[y:y+h, x+w-1-i]
                if np.sum(col) < h * 255 * 0.8:
                    if i > 0:
                        thicknesses.append(i)
                    break
            
            if thicknesses:
                # 중간값 사용 + 50% 안전 마진
                return int(np.median(thicknesses) * 1.5)
            else:
                return 70  # 기본값
        except:
            return 70
    
    def enhance_wedding_ring_complete(self, image, metal_type, lighting):
        """v13.3 완전한 10단계 보정 (대화 16-20)"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, 
                                                         self.params['champagne_gold']['natural'])
            
            # 1. 노이즈 제거
            image_denoised = cv2.bilateralFilter(image, 9, 75, 75)
            pil_image = Image.fromarray(image_denoised)
            
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
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(params.get('saturation', 1.0))
            
            # numpy 배열로 변환
            enhanced_array = np.array(enhanced)
            
            # 6. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(enhanced_array, 1 - params['white_overlay'],
                                           white_overlay, params['white_overlay'], 0)
            
            # 7. LAB 색공간에서 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            # 8. CLAHE (명료도)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9. 감마 보정
            gamma = params.get('gamma', 1.0)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_array = cv2.LUT(enhanced_array, table)
            
            # 10. 원본과 블렌딩
            final = cv2.addWeighted(enhanced_array, 1 - params['original_blend'],
                                  image, params['original_blend'], 0)
            
            return final
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
    
    def remove_black_border_perfect(self, image, mask, bbox, thickness, lighting):
        """검은색 선만 완벽히 제거 (대화 27번 v14.8 방식)"""
        try:
            x, y, w, h = bbox
            
            # 28쌍 AFTER 배경색 가져오기
            after_bg_color = np.array(self.after_bg_colors[lighting]['light'], dtype=np.uint8)
            
            # 검은색 선 마스크 생성 (두께 기반)
            border_mask = np.zeros_like(mask)
            
            # 각 변에 대해 실제 두께만큼만 마스킹
            border_mask[y:y+thickness, x:x+w] = 255  # 상단
            border_mask[y+h-thickness:y+h, x:x+w] = 255  # 하단
            border_mask[y:y+h, x:x+thickness] = 255  # 좌측
            border_mask[y:y+h, x+w-thickness:x+w] = 255  # 우측
            
            # 웨딩링 보호 영역 (안쪽 영역은 절대 건드리지 않음)
            inner_margin = thickness + 20
            inner_x = x + inner_margin
            inner_y = y + inner_margin
            inner_w = w - 2 * inner_margin
            inner_h = h - 2 * inner_margin
            
            # 웨딩링 영역 완전 보호
            if inner_w > 50 and inner_h > 50:
                border_mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = 0
            
            # 배경색으로 직접 교체 (대화 27번 핵심)
            result = image.copy()
            mask_indices = np.where(border_mask > 0)
            result[mask_indices] = after_bg_color
            
            # 부드러운 블렌딩 (31x31 가우시안 - 대화 25번)
            blurred_mask = cv2.GaussianBlur(border_mask.astype(np.float32), (31, 31), 10)
            blurred_mask = blurred_mask / 255.0
            
            # RGB 채널별로 블렌딩
            for c in range(3):
                result[:, :, c] = (image[:, :, c].astype(np.float32) * (1 - blurred_mask) +
                                  result[:, :, c].astype(np.float32) * blurred_mask).astype(np.uint8)
            
            return result, (inner_x, inner_y, inner_w, inner_h)
        except:
            return image, None
    
    def enhance_ring_area(self, image, ring_bbox):
        """웨딩링 영역 추가 보정 (대화 25번)"""
        try:
            if ring_bbox is None:
                return image
            
            x, y, w, h = ring_bbox
            
            # 영역 유효성 체크
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return image
            
            height, width = image.shape[:2]
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                return image
            
            result = image.copy()
            
            # 웨딩링 영역 추출
            ring_region = result[y:y+h, x:x+w].copy()
            
            # 추가 밝기 및 선명도 강화
            ring_enhanced = cv2.convertScaleAbs(ring_region, alpha=1.15, beta=10)
            
            # 언샤프 마스킹
            blurred = cv2.GaussianBlur(ring_enhanced, (0, 0), 2.0)
            ring_enhanced = cv2.addWeighted(ring_enhanced, 1.5, blurred, -0.5, 0)
            
            # 다시 적용
            result[y:y+h, x:x+w] = ring_enhanced
            
            return result
        except:
            return image
    
    def create_perfect_thumbnail_final(self, image, ring_bbox):
        """완벽한 1000x1300 썸네일 생성 (대화 27번)"""
        try:
            height, width = image.shape[:2]
            target_w, target_h = 1000, 1300
            
            if ring_bbox is not None:
                x, y, w, h = ring_bbox
                
                # 여백 최소화 (5% 마진만)
                margin = 0.05
                x1 = max(0, int(x - w * margin))
                y1 = max(0, int(y - h * margin))
                x2 = min(width, int(x + w * (1 + margin)))
                y2 = min(height, int(y + h * (1 + margin)))
                
                cropped = image[y1:y2, x1:x2]
            else:
                # 중앙 60% 영역 크롭
                margin = 0.2
                x1 = int(width * margin)
                y1 = int(height * margin)
                x2 = int(width * (1 - margin))
                y2 = int(height * (1 - margin))
                cropped = image[y1:y2, x1:x2]
            
            # 1000x1300 비율에 맞게 리사이즈
            crop_h, crop_w = cropped.shape[:2]
            
            # 웨딩링이 화면 가득 차도록 (대화 27번)
            ratio = max(target_w / crop_w, target_h / crop_h) * 0.98  # 98% 크기
            
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            # 고품질 리사이즈
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000x1300 캔버스에 중앙 배치
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            
            # 중앙 배치 (위아래 여백 최소화)
            start_x = max(0, (target_w - new_w) // 2)
            start_y = max(0, (target_h - new_h) // 2)
            
            end_x = min(target_w, start_x + new_w)
            end_y = min(target_h, start_y + new_h)
            
            # 캔버스에 맞게 조정
            if end_x - start_x != new_w or end_y - start_y != new_h:
                resized = resized[:end_y-start_y, :end_x-start_x]
            
            canvas[start_y:end_y, start_x:end_x] = resized
            
            return canvas
        except Exception as e:
            print(f"Thumbnail error: {e}")
            # 실패 시 전체 이미지 리사이즈
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event.get("input", {})
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "message": "v17.5 Ultimate - 완벽한 웨딩링 AI",
                "status": "ready",
                "version": "17.5-ultimate"
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            processor = UltimateWeddingRingProcessor()
            
            # 1. 검은색 테두리 감지 (전체 이미지에서)
            mask, bbox, thickness = processor.detect_black_border_complete(image_array)
            
            # 2. 금속 타입 및 조명 감지
            if mask is not None and bbox is not None:
                # 검은색 선 안쪽 영역에서 금속 감지
                x, y, w, h = bbox
                inner_mask = np.zeros_like(mask)
                margin = thickness + 20
                inner_mask[y+margin:y+h-margin, x+margin:x+w-margin] = 255
                metal_type = processor.detect_metal_type(image_array, inner_mask)
            else:
                metal_type = processor.detect_metal_type(image_array)
            
            lighting = processor.detect_lighting(image_array)
            
            # 3. v13.3 완전 보정 (무조건 실행)
            enhanced = processor.enhance_wedding_ring_complete(image_array, metal_type, lighting)
            
            # 4. 검은색 선 제거
            if mask is not None and bbox is not None and thickness > 0:
                # 검은색 선만 제거 (웨딩링 완전 보호)
                border_removed, ring_bbox = processor.remove_black_border_perfect(
                    enhanced, mask, bbox, thickness, lighting
                )
                
                # 5. 웨딩링 영역 추가 보정
                if ring_bbox is not None:
                    final_image = processor.enhance_ring_area(border_removed, ring_bbox)
                else:
                    final_image = border_removed
            else:
                # 검은색 선이 없어도 전체적으로 더 선명하게
                final_image = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=8)
                ring_bbox = None
            
            # 6. 2x 업스케일링
            height, width = final_image.shape[:2]
            upscaled = cv2.resize(final_image, (width * 2, height * 2), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            # 7. 썸네일 생성 (웨딩링 중심)
            if ring_bbox is not None:
                # 업스케일된 좌표로 변환
                thumb_bbox = (ring_bbox[0] * 2, ring_bbox[1] * 2, 
                             ring_bbox[2] * 2, ring_bbox[3] * 2)
            else:
                thumb_bbox = None
            
            thumbnail = processor.create_perfect_thumbnail_final(upscaled, thumb_bbox)
            
            # 8. 결과 인코딩
            # 메인 이미지
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "17.5-ultimate",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": bbox is not None,
                    "border_thickness": thickness,
                    "processing": "complete"
                }
            }
            
    except Exception as e:
        print(f"Processing error: {str(e)}")
        # 에러 시에도 기본 처리
        try:
            if "image_base64" in input_data:
                image_data = base64.b64decode(input_data["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image.convert('RGB'))
                
                # 최소한의 보정이라도 적용
                enhanced = cv2.convertScaleAbs(image_array, alpha=1.25, beta=15)
                
                # 샤프닝
                kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
                upscaled = cv2.resize(enhanced, (image_array.shape[1] * 2, image_array.shape[0] * 2))
                
                # 인코딩
                main_pil = Image.fromarray(upscaled)
                main_buffer = io.BytesIO()
                main_pil.save(main_buffer, format='JPEG', quality=95)
                main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
                
                # 썸네일
                thumbnail = cv2.resize(enhanced, (1000, 1300))
                thumb_pil = Image.fromarray(thumbnail)
                thumb_buffer = io.BytesIO()
                thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
                thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
                
                return {
                    "enhanced_image": main_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "version": "17.5-ultimate",
                        "error": str(e),
                        "fallback": True
                    }
                }
        except:
            pass
        
        return {"error": f"Processing failed: {str(e)}"}

# RunPod 실행
runpod.serverless.start({"handler": handler})
