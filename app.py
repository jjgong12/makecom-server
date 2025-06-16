import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 세트 (28쌍 학습 데이터 기반)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
            'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01,
            'clahe_clip': 1.15, 'noise_reduction': 1.1
        },
        'warm': {
            'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
            'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
            'original_blend': 0.18, 'saturation': 0.98, 'gamma': 1.03,
            'clahe_clip': 1.18, 'noise_reduction': 1.15
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.00,
            'clahe_clip': 1.20, 'noise_reduction': 1.08
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98,
            'clahe_clip': 1.10, 'noise_reduction': 1.05
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,  # 화이트화 강화
            'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,    # 화이트골드 방향
            'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00,   # 채도 감소
            'clahe_clip': 1.15, 'noise_reduction': 1.12
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01,
            'clahe_clip': 1.12, 'noise_reduction': 1.08
        }
    }
}

class EnhancedWeddingRingDetector:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS

    def detect_black_border_precise(self, image):
        """정밀한 검은색 테두리 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 가장자리에서만 검은색 테두리 찾기 (Dialog 34 웨딩링 보호)
            edge_width = 60  # 더 넓은 가장자리 영역
            edge_mask = np.zeros_like(gray)
            edge_mask[:edge_width, :] = 255  # 상단
            edge_mask[-edge_width:, :] = 255  # 하단
            edge_mask[:, :edge_width] = 255  # 좌측
            edge_mask[:, -edge_width:] = 255  # 우측
            
            # 다중 threshold로 정확한 검은색 감지
            best_contour = None
            best_area = 0
            
            for threshold in [10, 15, 20, 25, 30]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                border_only = cv2.bitwise_and(binary, edge_mask)
                
                # 형태학적 연산으로 정제
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                border_only = cv2.morphologyEx(border_only, cv2.MORPH_CLOSE, kernel)
                
                # 컨투어 찾기
                contours, _ = cv2.findContours(border_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 유효한 테두리 조건 (더 관대하게)
                    if (area > width * height * 0.05 and  # 최소 크기 줄임
                        w > width * 0.2 and h > height * 0.2 and  # 최소 크기 줄임
                        0.2 < w/h < 5.0):  # 비율 범위 확장
                        
                        if area > best_area:
                            best_area = area
                            best_contour = contour
            
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                return (x, y, w, h)
            
            return None
        except Exception as e:
            print(f"테두리 감지 오류: {e}")
            return None

    def detect_wedding_ring_inside_border(self, image, border_bbox=None):
        """검은색 테두리 내부에서 웨딩링 감지 - 핵심 강화!"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 검색 영역 설정
            if border_bbox is not None:
                x, y, w, h = border_bbox
                # 테두리 내부 중앙 70% 영역에서 웨딩링 찾기
                margin_w = int(w * 0.15)
                margin_h = int(h * 0.15)
                search_x = x + margin_w
                search_y = y + margin_h
                search_w = w - 2 * margin_w
                search_h = h - 2 * margin_h
                search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
                offset_x, offset_y = search_x, search_y
            else:
                # 전체 이미지에서 찾기
                search_region = gray
                offset_x, offset_y = 0, 0
                search_w, search_h = width, height
            
            if search_region.size == 0:
                return None
            
            # 방법 1: 밝은 영역 감지 (금속 반사)
            ring_candidates = []
            
            # 히스토그램 기반 밝은 영역 찾기
            hist = cv2.calcHist([search_region], [0], None, [256], [0, 256])
            bright_threshold = np.argmax(hist[100:]) + 100  # 밝은 부분의 peak
            bright_threshold = max(bright_threshold, np.mean(search_region) + np.std(search_region))
            
            _, bright_mask = cv2.threshold(search_region, bright_threshold, 255, cv2.THRESH_BINARY)
            
            # 형태학적 연산으로 링 모양 찾기
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_ellipse)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_ellipse)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # 링 모양 조건 (원형에 가까운)
                    if 0.3 <= aspect_ratio <= 3.0 and area > search_w * search_h * 0.01:
                        # 전역 좌표로 변환
                        global_x = offset_x + x
                        global_y = offset_y + y
                        ring_candidates.append((global_x, global_y, w, h, area))
            
            # 방법 2: 가장자리 감지 (링의 윤곽)
            edges = cv2.Canny(search_region, 30, 100)
            
            # 허프 원 변환으로 원형 감지
            circles = cv2.HoughCircles(search_region, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=30, minRadius=10, maxRadius=min(search_w, search_h)//3)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # 전역 좌표로 변환
                    global_x = offset_x + x - r
                    global_y = offset_y + y - r
                    ring_candidates.append((global_x, global_y, 2*r, 2*r, np.pi * r * r))
            
            # 방법 3: 색상 기반 금속 감지
            if border_bbox is not None:
                # LAB 색공간에서 금속 특성 찾기
                lab_region = cv2.cvtColor(image[offset_y:offset_y+search_h, offset_x:offset_x+search_w], cv2.COLOR_RGB2LAB)
                l_channel = lab_region[:, :, 0]
                
                # 밝은 금속 영역 찾기
                metal_threshold = np.mean(l_channel) + 0.5 * np.std(l_channel)
                _, metal_mask = cv2.threshold(l_channel, metal_threshold, 255, cv2.THRESH_BINARY)
                
                # 형태학적 연산
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        x, y, w, h = cv2.boundingRect(contour)
                        global_x = offset_x + x
                        global_y = offset_y + y
                        ring_candidates.append((global_x, global_y, w, h, area))
            
            # 최적의 웨딩링 후보 선택
            if ring_candidates:
                # 중앙에 가깝고 크기가 적당한 것 우선
                center_x, center_y = width // 2, height // 2
                
                def score_candidate(candidate):
                    x, y, w, h, area = candidate
                    candidate_center_x = x + w // 2
                    candidate_center_y = y + h // 2
                    
                    # 중앙으로부터의 거리
                    center_distance = np.sqrt((candidate_center_x - center_x)**2 + (candidate_center_y - center_y)**2)
                    max_distance = np.sqrt(width**2 + height**2)
                    center_score = 1 - (center_distance / max_distance)
                    
                    # 크기 점수 (너무 크거나 작지 않은)
                    ideal_size = min(width, height) * 0.3
                    size_diff = abs(max(w, h) - ideal_size)
                    size_score = 1 / (1 + size_diff / ideal_size)
                    
                    # 종횡비 점수 (정사각형에 가까운)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
                    aspect_score = 1 / aspect_ratio if aspect_ratio <= 2 else 1 / (aspect_ratio ** 2)
                    
                    return center_score * 0.4 + size_score * 0.4 + aspect_score * 0.2
                
                # 점수 계산하여 최적 후보 선택
                ring_candidates.sort(key=score_candidate, reverse=True)
                best_candidate = ring_candidates[0]
                
                return best_candidate[:4]  # (x, y, w, h)
            
            return None
        except Exception as e:
            print(f"웨딩링 감지 오류: {e}")
            return None

    def detect_wedding_ring_fallback(self, image):
        """검은색 마스킹 없이도 웨딩링 감지 (Fallback)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 중앙 영역 우선 검색
            center_x, center_y = width // 2, height // 2
            search_radius = min(width, height) // 3
            
            # 원형 마스크 생성
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= search_radius**2
            
            # 마스크 영역 내에서 밝은 부분 찾기
            masked_gray = gray.copy()
            masked_gray[~mask] = 0
            
            # 적응적 임계값으로 밝은 영역 찾기
            mean_val = np.mean(masked_gray[mask])
            std_val = np.std(masked_gray[mask])
            threshold = mean_val + 0.5 * std_val
            
            _, binary = cv2.threshold(masked_gray, threshold, 255, cv2.THRESH_BINARY)
            
            # 형태학적 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # 최소 크기
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return (x, y, w, h)
            
            # 그래도 못 찾으면 중앙 영역 반환
            fallback_size = min(width, height) // 4
            fallback_x = center_x - fallback_size // 2
            fallback_y = center_y - fallback_size // 2
            return (fallback_x, fallback_y, fallback_size, fallback_size)
            
        except Exception as e:
            print(f"Fallback 웨딩링 감지 오류: {e}")
            # 최후의 중앙 영역 반환
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            size = min(image.shape[0], image.shape[1]) // 4
            return (center_x - size//2, center_y - size//2, size, size)

    def enhance_wedding_ring_v13_3_complete(self, image, metal_type, lighting):
        """v13.3 완전한 10단계 보정"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, self.params['champagne_gold']['natural'])
            
            # 1. 노이즈 제거
            denoised = cv2.bilateralFilter(image, 9, int(80 * params.get('noise_reduction', 1.1)), 
                                         int(80 * params.get('noise_reduction', 1.1)))
            
            # 2. PIL 기반 기본 보정
            pil_image = Image.fromarray(denoised)
            
            # 3. 밝기 조정
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 4. 대비 조정  
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 5. 선명도 조정
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # 6. 채도 조정
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(params.get('saturation', 1.0))
            
            enhanced_array = np.array(enhanced)
            
            # 7. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            # 8. LAB 색공간 색온도 조정
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 9. CLAHE 명료도 향상
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=params.get('clahe_clip', 1.15), tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 10. 감마 보정
            gamma = params.get('gamma', 1.0)
            gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_array = cv2.LUT(enhanced_array, gamma_table)
            
            # 11. 원본과 블렌딩
            original_blend = params['original_blend']
            final = cv2.addWeighted(
                enhanced_array, 1 - original_blend,
                image, original_blend, 0
            )
            
            return final.astype(np.uint8)
        except Exception as e:
            print(f"v13.3 보정 오류: {e}")
            # Fallback 보정
            enhanced = cv2.convertScaleAbs(image, alpha=1.25, beta=15)
            return enhanced

    def remove_black_border_safe(self, image, border_bbox):
        """안전한 검은색 테두리 제거"""
        try:
            if border_bbox is None:
                return image
            
            height, width = image.shape[:2]
            x, y, w, h = border_bbox
            
            # 매우 보수적인 가장자리만 제거
            mask = np.zeros((height, width), dtype=np.uint8)
            edge_width = 25  # 가장자리만 좁게
            
            # 상하좌우 가장자리만
            mask[y:y+edge_width, x:x+w] = 255  # 상단
            mask[y+h-edge_width:y+h, x:x+w] = 255  # 하단
            mask[y:y+h, x:x+edge_width] = 255  # 좌측
            mask[y:y+h, x+w-edge_width:x+w] = 255  # 우측
            
            # TELEA inpainting
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            
            # 부드러운 블렌딩
            smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 7)
            smooth_mask = smooth_mask / 255.0
            
            result = image.copy()
            for c in range(3):
                result[:, :, c] = (
                    image[:, :, c].astype(np.float32) * (1 - smooth_mask) +
                    inpainted[:, :, c].astype(np.float32) * smooth_mask
                )
            
            return result.astype(np.uint8)
        except Exception as e:
            print(f"테두리 제거 오류: {e}")
            return image

    def create_perfect_thumbnail_enhanced(self, image, ring_bbox, target_size=(1000, 1300)):
        """강화된 썸네일 생성 - 핵심 개선!"""
        try:
            height, width = image.shape[:2]
            target_w, target_h = target_size
            
            if ring_bbox is not None:
                # 웨딩링 기준으로 크롭
                rx, ry, rw, rh = ring_bbox
                center_x = rx + rw // 2
                center_y = ry + rh // 2
                
                # 웨딩링 크기의 3배 영역으로 크롭 (충분한 여백)
                crop_size = max(rw, rh) * 3
                print(f"웨딩링 기준 크롭: center=({center_x}, {center_y}), size={crop_size}")
            else:
                # Fallback: 중앙 기준
                center_x, center_y = width // 2, height // 2
                crop_size = min(width, height) // 2
                print(f"중앙 기준 크롭: center=({center_x}, {center_y}), size={crop_size}")
            
            # 크롭 영역 계산
            half_crop = crop_size // 2
            x1 = max(0, center_x - half_crop)
            y1 = max(0, center_y - half_crop)
            x2 = min(width, center_x + half_crop)
            y2 = min(height, center_y + half_crop)
            
            # 실제 크롭 실행
            cropped = image[y1:y2, x1:x2]
            print(f"크롭된 이미지 크기: {cropped.shape}")
            
            if cropped.size == 0:
                print("크롭 실패, 전체 이미지 사용")
                cropped = image
            
            # 1000×1300 비율로 리사이즈
            crop_h, crop_w = cropped.shape[:2]
            
            # 웨딩링이 화면을 많이 차지하도록 (85% 크기)
            ratio = max(target_w / crop_w, target_h / crop_h) * 0.85
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            print(f"리사이즈: {crop_w}x{crop_h} -> {new_w}x{new_h}")
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                print("리사이즈 실패, 원본 사용")
                resized = cropped
            
            # 1000×1300 캔버스에 배치
            canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
            
            # 중앙 배치 (약간 위쪽으로)
            start_x = max(0, (target_w - new_w) // 2)
            start_y = max(0, (target_h - new_h) // 3)  # 1/3 지점 (위쪽으로)
            
            # 범위 확인하여 안전하게 배치
            end_x = min(target_w, start_x + new_w)
            end_y = min(target_h, start_y + new_h)
            actual_w = end_x - start_x
            actual_h = end_y - start_y
            
            if actual_w > 0 and actual_h > 0:
                canvas[start_y:end_y, start_x:end_x] = resized[:actual_h, :actual_w]
                print(f"캔버스 배치 성공: ({start_x}, {start_y}) - ({end_x}, {end_y})")
            else:
                print("캔버스 배치 실패")
            
            return canvas
            
        except Exception as e:
            print(f"썸네일 생성 오류: {e}")
            # 최후의 Fallback
            try:
                center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                crop_size = min(image.shape[0], image.shape[1]) // 3
                y1, y2 = max(0, center_y - crop_size), min(image.shape[0], center_y + crop_size)
                x1, x2 = max(0, center_x - crop_size), min(image.shape[1], center_x + crop_size)
                cropped = image[y1:y2, x1:x2]
                return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            except:
                # 진짜 최후의 수단
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def handler(event):
    """RunPod Serverless 메인 핸들러 - 강화된 웨딩링 감지"""
    try:
        input_data = event["input"]
        
        # 기본 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"v17.2 Enhanced Detection 연결 성공: {input_data['prompt']}",
                "status": "enhanced_ring_detection_ready",
                "version": "v17.2 - 강화된 웨딩링 감지 시스템",
                "capabilities": ["정밀 웨딩링 감지", "검은색 테두리 내부 감지", "Fallback 감지", "완벽한 썸네일"]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            detector = EnhancedWeddingRingDetector()
            
            # 1. 검은색 테두리 감지
            border_bbox = detector.detect_black_border_precise(image_array)
            print(f"검은색 테두리: {border_bbox}")
            
            # 2. 웨딩링 감지 (강화된 시스템)
            if border_bbox is not None:
                # 테두리 내부에서 웨딩링 찾기
                ring_bbox = detector.detect_wedding_ring_inside_border(image_array, border_bbox)
                print(f"테두리 내부 웨딩링: {ring_bbox}")
            else:
                ring_bbox = None
            
            # 3. Fallback 웨딩링 감지
            if ring_bbox is None:
                ring_bbox = detector.detect_wedding_ring_fallback(image_array)
                print(f"Fallback 웨딩링: {ring_bbox}")
            
            # 4. 금속/조명 감지
            if ring_bbox is not None:
                rx, ry, rw, rh = ring_bbox
                ring_region = image_array[ry:ry+rh, rx:rx+rw]
                if ring_region.size > 0:
                    # 웨딩링 영역 기준으로 금속 감지
                    hsv_ring = cv2.cvtColor(ring_region, cv2.COLOR_RGB2HSV)
                    avg_hue = np.mean(hsv_ring[:, :, 0])
                    avg_sat = np.mean(hsv_ring[:, :, 1])
                    
                    if avg_sat < 30:
                        metal_type = 'white_gold'
                    elif 5 <= avg_hue <= 25:
                        metal_type = 'champagne_gold'  # 기본값
                    else:
                        metal_type = 'champagne_gold'
                else:
                    metal_type = 'champagne_gold'
            else:
                metal_type = 'champagne_gold'
            
            lighting = 'natural'  # 기본값
            
            # 5. v13.3 완전한 보정 (무조건 실행)
            enhanced_image = detector.enhance_wedding_ring_v13_3_complete(image_array, metal_type, lighting)
            
            # 6. 검은색 테두리 제거 (있으면)
            if border_bbox is not None:
                enhanced_image = detector.remove_black_border_safe(enhanced_image, border_bbox)
            
            # 7. 웨딩링 영역 추가 강화
            if ring_bbox is not None:
                rx, ry, rw, rh = ring_bbox
                if rx >= 0 and ry >= 0 and rx + rw <= enhanced_image.shape[1] and ry + rh <= enhanced_image.shape[0]:
                    ring_region = enhanced_image[ry:ry+rh, rx:rx+rw].copy()
                    # 추가 밝기와 선명도
                    ring_enhanced = cv2.convertScaleAbs(ring_region, alpha=1.15, beta=10)
                    enhanced_image[ry:ry+rh, rx:rx+rw] = ring_enhanced
            
            # 8. 2x 업스케일링
            upscaled_image = cv2.resize(enhanced_image, 
                                      (enhanced_image.shape[1] * 2, enhanced_image.shape[0] * 2), 
                                      interpolation=cv2.INTER_LANCZOS4)
            
            # 업스케일된 좌표 조정
            if ring_bbox is not None:
                ring_bbox_scaled = tuple(coord * 2 for coord in ring_bbox)
            else:
                ring_bbox_scaled = None
            
            # 9. 강화된 썸네일 생성
            thumbnail = detector.create_perfect_thumbnail_enhanced(upscaled_image, ring_bbox_scaled)
            
            # 10. 결과 인코딩
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
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v17.2 Enhanced Detection",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_detected": border_bbox is not None,
                    "ring_detected": ring_bbox is not None,
                    "ring_bbox": ring_bbox,
                    "detection_method": "테두리 내부 감지" if border_bbox and ring_bbox else "Fallback 감지",
                    "v13_3_applied": True,
                    "thumbnail_method": "웨딩링 중심" if ring_bbox else "중앙 크롭",
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}",
                    "thumbnail_size": "1000x1300"
                }
            }
    
    except Exception as e:
        print(f"메인 처리 오류: {e}")
        # Emergency 처리
        try:
            if "image_base64" in input_data:
                image_data = base64.b64decode(input_data["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image.convert('RGB'))
                
                # 기본 보정
                enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=20)
                upscaled = cv2.resize(enhanced, (enhanced.shape[1] * 2, enhanced.shape[0] * 2), 
                                    interpolation=cv2.INTER_LANCZOS4)
                
                # 중앙 썸네일
                center_y, center_x = upscaled.shape[0] // 2, upscaled.shape[1] // 2
                crop_size = min(upscaled.shape[0], upscaled.shape[1]) // 3
                y1, y2 = center_y - crop_size, center_y + crop_size
                x1, x2 = center_x - crop_size, center_x + crop_size
                thumbnail = cv2.resize(upscaled[y1:y2, x1:x2], (1000, 1300), interpolation=cv2.INTER_LANCZOS4)
                
                # 인코딩
                main_pil = Image.fromarray(upscaled)
                main_buffer = io.BytesIO()
                main_pil.save(main_buffer, format='JPEG', quality=95)
                main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
                
                thumb_pil = Image.fromarray(thumbnail)
                thumb_buffer = io.BytesIO()
                thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
                thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
                
                return {
                    "enhanced_image": main_base64,
                    "thumbnail": thumb_base64,
                    "processing_info": {
                        "version": "v17.2 Emergency",
                        "error": str(e),
                        "basic_processing": True
                    }
                }
        except:
            pass
        
        return {"error": f"v17.2 처리 중 오류 발생: {str(e)}"}

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
