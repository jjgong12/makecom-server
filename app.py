import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반)
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
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.10,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.20, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.15,
            'sharpness': 1.25, 'color_temp_a': -3, 'color_temp_b': -3,
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

# 28쌍 AFTER 파일들의 배경색 (대화 28번)
AFTER_BACKGROUND_COLORS = {
    'natural': {'light': [250, 248, 245], 'medium': [242, 240, 237]},
    'warm': {'light': [252, 250, 245], 'medium': [245, 242, 237]},
    'cool': {'light': [248, 250, 252], 'medium': [240, 242, 245]}
}

class WeddingRingProcessor:
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
            
            # 금속 타입 분류
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
    
    def detect_actual_line_thickness(self, mask, bbox):
        """실제 검은색 선 두께 측정 (대화 29번 핵심)"""
        try:
            x, y, w, h = bbox
            
            # 4방향에서 실제 선 두께 측정
            thicknesses = []
            
            # 상단 선
            top_line = mask[y:y+min(150, h//3), x:x+w]
            if np.any(top_line):
                for row in range(top_line.shape[0]):
                    if np.sum(top_line[row]) < w * 255 * 0.8:
                        break
                if row > 0:
                    thicknesses.append(row)
            
            # 하단 선
            bottom_line = mask[max(y+h-150, y+h*2//3):y+h, x:x+w]
            if np.any(bottom_line):
                for row in range(bottom_line.shape[0]-1, -1, -1):
                    if np.sum(bottom_line[row]) < w * 255 * 0.8:
                        break
                if row < bottom_line.shape[0]-1:
                    thicknesses.append(bottom_line.shape[0]-1-row)
            
            # 좌측 선
            left_line = mask[y:y+h, x:x+min(150, w//3)]
            if np.any(left_line):
                for col in range(left_line.shape[1]):
                    if np.sum(left_line[:, col]) < h * 255 * 0.8:
                        break
                if col > 0:
                    thicknesses.append(col)
            
            # 우측 선
            right_line = mask[y:y+h, max(x+w-150, x+w*2//3):x+w]
            if np.any(right_line):
                for col in range(right_line.shape[1]-1, -1, -1):
                    if np.sum(right_line[:, col]) < h * 255 * 0.8:
                        break
                if col < right_line.shape[1]-1:
                    thicknesses.append(right_line.shape[1]-1-col)
            
            if thicknesses:
                # 중간값 사용 (안정성)
                thickness = int(np.median(thicknesses))
                # 50% 오차 범위 추가
                return int(thickness * 1.5)
            else:
                return 50  # 기본값
        except:
            return 50
    
    def detect_black_border_safe(self, image):
        """가장자리에서만 검은색 선 감지 (대화 34번 핵심)"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 가장자리 50픽셀에서만 검은색 선 찾기 (중앙 웨딩링 보호)
            edge_mask = np.zeros_like(gray)
            edge_width = 50
            edge_mask[:edge_width, :] = 255  # 상단
            edge_mask[-edge_width:, :] = 255  # 하단
            edge_mask[:, :edge_width] = 255  # 좌측
            edge_mask[:, -edge_width:] = 255  # 우측
            
            # 다중 threshold로 검은색 감지 (20, 30, 40)
            combined_mask = np.zeros_like(gray)
            for threshold in [20, 30, 40]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                border_only = cv2.bitwise_and(binary, edge_mask)
                combined_mask = cv2.bitwise_or(combined_mask, border_only)
            
            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # 가장 큰 사각형 컨투어 찾기
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 사각형인지 확인
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                
                if len(approx) >= 4:  # 사각형 형태
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 이미지 크기의 30% 이상인 경우만 유효한 테두리로 간주
                    if w > width * 0.3 and h > height * 0.3:
                        return combined_mask, (x, y, w, h)
            
            return None, None
        except:
            return None, None
    
    def get_after_background_color(self, lighting):
        """28쌍 AFTER 파일들의 배경색 가져오기"""
        colors = self.after_bg_colors.get(lighting, self.after_bg_colors['natural'])
        return np.array(colors['light'], dtype=np.uint8)
    
    def remove_black_border_v17_5(self, image, mask, bbox, lighting):
        """검은색 선만 제거하고 웨딩링은 완전 보호 (최종 완성)"""
        try:
            x, y, w, h = bbox
            height, width = image.shape[:2]
            
            # 실제 선 두께 측정
            actual_thickness = self.detect_actual_line_thickness(mask, bbox)
            
            # 웨딩링 보호 영역 설정 (매우 보수적)
            margin = max(actual_thickness + 30, int(min(w, h) * 0.15))
            
            protected_x = x + margin
            protected_y = y + margin
            protected_w = w - 2 * margin
            protected_h = h - 2 * margin
            
            # 웨딩링 영역이 너무 작아지지 않도록 보장
            min_size = min(width, height) // 3
            if protected_w < min_size or protected_h < min_size:
                protected_x = width // 4
                protected_y = height // 4
                protected_w = width // 2
                protected_h = height // 2
            
            # 가장자리만 제거할 마스크 생성
            edge_only_mask = np.zeros_like(mask)
            edge_thickness = min(actual_thickness, 50)  # 최대 50픽셀
            
            # 상하좌우 가장자리만 마킹
            edge_only_mask[y:y+edge_thickness, x:x+w] = 255  # 상단
            edge_only_mask[y+h-edge_thickness:y+h, x:x+w] = 255  # 하단
            edge_only_mask[y:y+h, x:x+edge_thickness] = 255  # 좌측
            edge_only_mask[y:y+h, x+w-edge_thickness:x+w] = 255  # 우측
            
            # 웨딩링 보호 영역은 절대 제거하지 않음
            edge_only_mask[protected_y:protected_y+protected_h, 
                          protected_x:protected_x+protected_w] = 0
            
            # 28쌍 AFTER 배경색 가져오기
            after_bg_color = self.get_after_background_color(lighting)
            
            # 배경색으로 직접 교체 (대화 27번 v14.8 방식)
            result = image.copy()
            mask_indices = np.where(edge_only_mask > 0)
            result[mask_indices] = after_bg_color
            
            # 부드러운 블렌딩 (31x31 가우시안)
            blurred_mask = cv2.GaussianBlur(edge_only_mask.astype(np.float32), (31, 31), 10)
            blurred_mask = blurred_mask / 255.0
            
            for c in range(3):
                result[:, :, c] = (image[:, :, c].astype(np.float32) * (1 - blurred_mask) +
                                  result[:, :, c].astype(np.float32) * blurred_mask).astype(np.uint8)
            
            return result, (protected_x, protected_y, protected_w, protected_h)
        except:
            return image, None
    
    def enhance_wedding_ring_v13_3(self, image, metal_type, lighting):
        """v13.3 완전한 10단계 보정"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, 
                                                         self.params['champagne_gold']['natural'])
            
            # PIL로 변환
            pil_image = Image.fromarray(image)
            
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
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
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
    
    def create_perfect_thumbnail(self, image, bbox=None):
        """완벽한 1000x1300 썸네일 생성 (여백 최소화)"""
        try:
            height, width = image.shape[:2]
            target_w, target_h = 1000, 1300
            
            if bbox is not None:
                # 웨딩링 영역 기준 크롭
                x, y, w, h = bbox
                
                # 여백 최소화 (10% 마진)
                margin_w = int(w * 0.1)
                margin_h = int(h * 0.1)
                
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h)
                x2 = min(width, x + w + margin_w)
                y2 = min(height, y + h + margin_h)
                
                cropped = image[y1:y2, x1:x2]
            else:
                # 중앙 영역에서 밝은 부분(웨딩링) 찾기
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                center_x, center_y = width // 2, height // 2
                search_w, search_h = width // 2, height // 2
                
                center_region = gray[center_y-search_h//2:center_y+search_h//2,
                                   center_x-search_w//2:center_x+search_w//2]
                
                threshold = np.mean(center_region) + np.std(center_region)
                _, binary = cv2.threshold(center_region, threshold, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 절대 좌표로 변환
                    x += center_x - search_w // 2
                    y += center_y - search_h // 2
                    
                    # 10% 마진으로 크롭
                    margin = 0.1
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
            ratio = max(target_w / crop_w, target_h / crop_h) * 0.95  # 95% 크기로 (약간의 여백)
            
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1000x1300 캔버스에 중앙 배치
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            
            # 위쪽에 더 가깝게 배치 (위아래 여백 줄이기)
            start_x = (target_w - new_w) // 2
            start_y = max(0, (target_h - new_h) // 4)  # 1/4 지점에 배치
            
            end_x = start_x + new_w
            end_y = start_y + new_h
            
            if end_y > target_h:
                end_y = target_h
                start_y = target_h - new_h
            
            canvas[start_y:end_y, start_x:end_x] = resized
            
            return canvas
        except:
            # 실패 시 원본 리사이즈
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event.get("input", {})
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "message": "v17.5 Final - 모든 문제 해결 완전체",
                "status": "ready",
                "version": "17.5"
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            processor = WeddingRingProcessor()
            
            # 1. 금속 타입 및 조명 감지
            metal_type = processor.detect_metal_type(image_array)
            lighting = processor.detect_lighting(image_array)
            
            # 2. v13.3 완전 보정 (무조건 실행)
            enhanced = processor.enhance_wedding_ring_v13_3(image_array, metal_type, lighting)
            
            # 3. 가장자리에서만 검은색 선 감지
            mask, bbox = processor.detect_black_border_safe(enhanced)
            
            if mask is not None and bbox is not None:
                # 4. 검은색 선만 제거 (웨딩링 완전 보호)
                border_removed, ring_bbox = processor.remove_black_border_v17_5(enhanced, mask, bbox, lighting)
                
                # 5. 웨딩링 영역 추가 보정
                if ring_bbox is not None:
                    rx, ry, rw, rh = ring_bbox
                    ring_region = border_removed[ry:ry+rh, rx:rx+rw].copy()
                    
                    # 웨딩링 더 밝고 선명하게
                    ring_enhanced = cv2.convertScaleAbs(ring_region, alpha=1.15, beta=10)
                    
                    # 언샤프 마스킹
                    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                    ring_enhanced = cv2.filter2D(ring_enhanced, -1, kernel)
                    
                    border_removed[ry:ry+rh, rx:rx+rw] = ring_enhanced
                    
                    final_image = border_removed
                else:
                    final_image = border_removed
            else:
                # 검은색 선이 없어도 전체적으로 더 선명하게
                final_image = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            # 6. 2x 업스케일링
            height, width = final_image.shape[:2]
            upscaled = cv2.resize(final_image, (width * 2, height * 2), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            # 7. 썸네일 생성 (웨딩링 중심)
            thumbnail_bbox = None
            if bbox is not None and mask is not None:
                # 검은색 선 안쪽 영역 기준
                x, y, w, h = bbox
                thickness = processor.detect_actual_line_thickness(mask, bbox)
                margin = thickness + 20
                
                thumb_x = x + margin
                thumb_y = y + margin
                thumb_w = w - 2 * margin
                thumb_h = h - 2 * margin
                
                if thumb_w > 50 and thumb_h > 50:
                    thumbnail_bbox = (thumb_x * 2, thumb_y * 2, thumb_w * 2, thumb_h * 2)
            
            thumbnail = processor.create_perfect_thumbnail(upscaled, thumbnail_bbox)
            
            # 8. 결과 인코딩
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
                    "version": "17.5",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": bbox is not None,
                    "border_thickness": processor.detect_actual_line_thickness(mask, bbox) if mask is not None and bbox is not None else 0,
                    "processing": "complete"
                }
            }
            
    except Exception as e:
        # 에러 시에도 기본 처리
        try:
            if "image_base64" in input_data:
                image_data = base64.b64decode(input_data["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image.convert('RGB'))
                
                # 최소한의 보정이라도 적용
                enhanced = cv2.convertScaleAbs(image_array, alpha=1.2, beta=10)
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
                        "version": "17.5",
                        "error": str(e),
                        "fallback": True
                    }
                }
        except:
            pass
        
        return {"error": f"Processing failed: {str(e)}"}

# RunPod 실행
runpod.serverless.start({"handler": handler})
