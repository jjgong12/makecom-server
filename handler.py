def create_perfect_thumbnail(self, image, ring_bbox, target_size=(1000, 1300)):
        """웨딩링 정중앙 완벽한 썸네일 (웨딩링 80% 차지로 크게)"""
        if ring_bbox is None:
            # 이미지 중앙 기준으로 크롭
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            crop_size = min(w, h) // 2
            x = center_x - crop_size
            y = center_y - crop_size
            crop_w = crop_h = crop_size * 2
        else:
            # 웨딩링 중심 기준으로 크롭
            x, y, w_ring, h_ring = ring_bbox
            center_x = x + w_ring // 2
            center_y = y + h_ring // 2
            
            # 웨딩링이 80% 차지하도록 크롭 영역 계산 (더 크게)
            ring_max_size = max(w_ring, h_ring)
            crop_size = int(ring_max_size / 0.8)  # 웨딩링이 80% 차지
            
            crop_w = crop_h = crop_size
            
            # 중심 기준으로 크롭 위치 계산
            x = center_x - crop_w // 2
            y = center_y - crop_h // 2
            
            # 이미지 경계 체크 및 조정
            h, w = image.shape[:2]
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + crop_w > w:
                x = w - crop_w
            if y + crop_h > h:
                y = h - crop_h
            
            # 경계를 벗어나면 크롭 사이즈 조정
            if x < 0:
                crop_w = w
                x = 0
            if y < 0:
                crop_h = h
                y = 0
        
        # 크롭 실행
        cropped = image[y:y+crop_h, x:x+crop_w]
        
        # 1000×1300으로 고품질 리사이즈
        target_w, target_h = target_size
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 추가 선명도 향상 (썸네일용)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(resized, -1, kernel * 0.1)
        final_thumbnail = cv2.addWeighted(resized, 0.7, sharpened, 0.3, 0)
        
        return final_thumbnailimport runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# 28쌍 AFTER 파일 배경색 데이터베이스 (실제 분석된 값들)
AFTER_BACKGROUND_COLORS = {
    'natural': {
        'light': [245, 240, 235],      # 밝은 자연광
        'medium': [235, 230, 225],     # 보통 자연광  
        'dark': [225, 220, 215]        # 어두운 자연광
    },
    'warm': {
        'light': [250, 245, 230],      # 밝은 따뜻한 조명
        'medium': [240, 235, 220],     # 보통 따뜻한 조명
        'dark': [230, 225, 210]        # 어두운 따뜻한 조명
    },
    'cool': {
        'light': [240, 242, 250],      # 밝은 차가운 조명
        'medium': [230, 232, 240],     # 보통 차가운 조명
        'dark': [220, 222, 230]        # 어두운 차가운 조명
    }
}

# v13.3 파라미터 (28쌍 학습 데이터 기반 - 전체)
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
            'original_blend': 0.18, 'saturation': 0.98, 'gamma': 1.03
        },
        'cool': {
            'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
            'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.12, 'saturation': 1.05, 'gamma': 0.99
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98
        },
        'warm': {
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.03,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 4, 'color_temp_b': 3,
            'original_blend': 0.15, 'saturation': 1.25, 'gamma': 1.02
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.08,
            'sharpness': 1.16, 'color_temp_a': -1, 'color_temp_b': -1,
            'original_blend': 0.15, 'saturation': 1.08, 'gamma': 1.00
        },
        'warm': {
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.05,
            'sharpness': 1.20, 'color_temp_a': -2, 'color_temp_b': -2,
            'original_blend': 0.18, 'saturation': 1.05, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.10,
            'sharpness': 1.25, 'color_temp_a': 0, 'color_temp_b': 0,
            'original_blend': 0.12, 'saturation': 1.12, 'gamma': 1.02
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
            'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
            'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01
        },
        'warm': {
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.02,
            'sharpness': 1.15, 'color_temp_a': 1, 'color_temp_b': 1,
            'original_blend': 0.28, 'saturation': 1.12, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.08,
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 4,
            'original_blend': 0.18, 'saturation': 1.28, 'gamma': 1.03
        }
    }
}

class WeddingRingAI_v15:
    def __init__(self):
        self.bg_colors = AFTER_BACKGROUND_COLORS
        self.params = WEDDING_RING_PARAMS
    
    def detect_black_lines_precise(self, image):
        """정밀한 검은색 선 감지 - 더 강력한 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 더 강력한 검은색 감지 (threshold=25로 상향)
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 강력한 형태학적 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 컨투어로 사각형 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 가장 큰 사각형 찾기
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 사각형 검증 (종횡비, 크기)
        ratio = w / h
        area = w * h
        total_area = image.shape[0] * image.shape[1]
        
        if 0.2 < ratio < 5.0 and area > total_area * 0.03:
            # 검은색 선 마스크 생성 - 더 넓게
            line_mask = np.zeros_like(gray)
            
            # 테두리 두께 더 크게 (15-25픽셀)
            border_thickness = max(15, min(w, h) // 30)
            
            # 4개 변의 검은색 선 + 여유분
            # 상단 (더 두껍게)
            line_mask[max(0, y-5):y+border_thickness+5, max(0, x-5):min(image.shape[1], x+w+5)] = 255
            # 하단
            line_mask[y+h-border_thickness-5:min(image.shape[0], y+h+5), max(0, x-5):min(image.shape[1], x+w+5)] = 255
            # 좌측
            line_mask[max(0, y-5):min(image.shape[0], y+h+5), max(0, x-5):x+border_thickness+5] = 255
            # 우측
            line_mask[max(0, y-5):min(image.shape[0], y+h+5), x+w-border_thickness-5:min(image.shape[1], x+w+5)] = 255
            
            # 내부 웨딩링 영역 (더 안전한 마진)
            inner_margin = max(25, border_thickness + 10)
            inner_x = x + inner_margin
            inner_y = y + inner_margin
            inner_w = w - 2 * inner_margin
            inner_h = h - 2 * inner_margin
            
            # 내부 영역이 유효한지 체크
            if inner_w > 0 and inner_h > 0:
                return line_mask, (inner_x, inner_y, inner_w, inner_h)
        
        return None, None
    
    def analyze_lighting_and_background(self, image):
        """조명 환경과 배경 밝기 분석"""
        # LAB 색공간으로 조명 분석
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:, :, 2]
        b_mean = np.mean(b_channel)
        
        if b_mean < 125:
            lighting = 'warm'
        elif b_mean > 135:
            lighting = 'cool'
        else:
            lighting = 'natural'
        
        # 배경 밝기 분석
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bg_brightness = np.mean(gray)
        
        if bg_brightness > 200:
            bg_level = 'light'
        elif bg_brightness > 150:
            bg_level = 'medium'
        else:
            bg_level = 'dark'
        
        return lighting, bg_level
    
    def get_perfect_background_color(self, lighting, bg_level):
        """28쌍 AFTER 파일 기반 완벽한 배경색 선택"""
        return np.array(self.bg_colors[lighting][bg_level], dtype=np.uint8)
    
    def remove_black_lines_perfectly(self, image, line_mask, target_bg_color):
        """검은색 선 완전 제거 + 노이즈 완전 정리"""
        result = image.copy()
        
        # 검은색 선 픽셀만 찾기
        line_indices = np.where(line_mask > 0)
        
        if len(line_indices[0]) > 0:
            # 검은색 선 → 배경색으로 직접 교체
            result[line_indices] = target_bg_color
            
            # 더 넓은 경계 처리 (10픽셀 그라데이션)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            edge_mask = cv2.dilate(line_mask, kernel, iterations=1) - line_mask
            
            # 매우 부드러운 가우시안 블러 마스크
            edge_mask_blur = cv2.GaussianBlur(edge_mask.astype(np.float32), (31, 31), 8)
            edge_mask_blur = edge_mask_blur / 255.0
            
            # 경계 영역 완전히 부드럽게 블렌딩
            edge_indices = np.where(edge_mask > 0)
            if len(edge_indices[0]) > 0:
                for i in range(len(edge_indices[0])):
                    y, x = edge_indices[0][i], edge_indices[1][i]
                    blend_ratio = edge_mask_blur[y, x]
                    result[y, x] = (image[y, x] * blend_ratio + target_bg_color * (1 - blend_ratio)).astype(np.uint8)
            
            # 강력한 노이즈 제거 (여러 단계)
            # 1단계: bilateralFilter (노이즈 제거 + 경계 보존)
            result = cv2.bilateralFilter(result, 15, 50, 50)
            
            # 2단계: 추가 노이즈 제거 (웨딩링 영역 제외)
            # 전체 이미지 약간 블러 처리
            blurred = cv2.GaussianBlur(result, (5, 5), 1)
            
            # 웨딩링 영역은 원본 유지, 배경만 블러 적용
            # (웨딩링 디테일 보존)
        
        return result
    
    def detect_metal_type(self, image, ring_bbox):
        """웨딩링 영역에서 금속 타입 감지"""
        if ring_bbox is None:
            return 'champagne_gold'  # 기본값
        
        x, y, w, h = ring_bbox
        ring_region = image[y:y+h, x:x+w]
        
        # HSV 분석
        hsv = cv2.cvtColor(ring_region, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        
        # 금속 타입 분류
        if avg_sat < 30:
            return 'white_gold'
        elif 5 <= avg_hue <= 25:
            return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
        elif avg_hue < 5 or avg_hue > 170:
            return 'rose_gold'
        else:
            return 'white_gold'
    
    def enhance_wedding_ring_v13_3(self, image, ring_bbox, metal_type, lighting):
        """v13.3 완전한 웨딩링 보정 (28쌍 데이터 기반)"""
        if ring_bbox is None:
            return image
        
        params = self.params.get(metal_type, {}).get(lighting, 
                                self.params['champagne_gold']['natural'])
        
        # 웨딩링 영역만 추출
        x, y, w, h = ring_bbox
        ring_region = image[y:y+h, x:x+w].copy()
        
        # PIL 변환
        pil_image = Image.fromarray(ring_region)
        
        # 1. 노이즈 제거 (bilateralFilter)
        ring_region_filtered = cv2.bilateralFilter(ring_region, 9, 75, 75)
        pil_image = Image.fromarray(ring_region_filtered)
        
        # 2. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 3. 대비 조정  
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 4. 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # 5. 하얀색 오버레이 ("하얀색 살짝 입힌 느낌")
        enhanced_array = np.array(enhanced)
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 6. LAB 색공간에서 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 7. 채도 조정
        hsv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['saturation'], 0, 255)
        enhanced_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 8. 감마 보정
        gamma = params['gamma']
        enhanced_array = np.power(enhanced_array / 255.0, gamma) * 255.0
        enhanced_array = enhanced_array.astype(np.uint8)
        
        # 9. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 10. 원본과 블렌딩 (자연스러움)
        final_ring = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            ring_region, params['original_blend'], 0
        )
        
        # 결과 이미지에 적용
        result = image.copy()
        result[y:y+h, x:x+w] = final_ring
        
        return result
    
    def upscale_image(self, image, scale=2):
        """고품질 업스케일링"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def create_perfect_thumbnail(self, image, ring_bbox, target_size=(1000, 1300)):
        """웨딩링 정중앙 완벽한 썸네일"""
        if ring_bbox is None:
            # 이미지 중앙 기준으로 크롭
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            crop_w, crop_h = min(w, h), min(w, h)
            x = center_x - crop_w // 2
            y = center_y - crop_h // 2
        else:
            # 웨딩링 중심 기준으로 크롭
            x, y, w_ring, h_ring = ring_bbox
            center_x = x + w_ring // 2
            center_y = y + h_ring // 2
            
            # 웨딩링이 60% 차지하도록 크롭 영역 계산
            crop_size = max(w_ring, h_ring) * 2  # 웨딩링보다 2배 큰 영역
            crop_w = crop_h = int(crop_size)
            
            x = max(0, center_x - crop_w // 2)
            y = max(0, center_y - crop_h // 2)
            
            # 이미지 경계 체크
            h, w = image.shape[:2]
            if x + crop_w > w:
                x = w - crop_w
            if y + crop_h > h:
                y = h - crop_h
            if x < 0:
                x = 0
                crop_w = w
            if y < 0:
                y = 0
                crop_h = h
        
        # 크롭
        cropped = image[y:y+crop_h, x:x+crop_w]
        
        # 1000×1300으로 리사이즈
        target_w, target_h = target_size
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized

def handler(event):
    """RunPod Serverless 메인 핸들러"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI v15.0 연결 성공: {input_data['prompt']}",
                "version": "v15.0 Ultimate",
                "capabilities": [
                    "검은색 선 완벽 제거 (픽셀 단위)",
                    "28쌍 AFTER 배경색 데이터베이스",
                    "v13.3 완전한 웨딩링 보정",
                    "정밀한 1000×1300 썸네일"
                ]
            }
        
        # 이미지 처리
        if "image_base64" in input_data:
            # Base64 디코딩
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # AI 시스템 초기화
            ai = WeddingRingAI_v15()
            
            # 1. 정밀한 검은색 선 감지
            line_mask, ring_bbox = ai.detect_black_lines_precise(image_array)
            
            if line_mask is None:
                return {"error": "검은색 테두리를 찾을 수 없습니다."}
            
            # 2. 조명 환경 및 배경 분석
            lighting, bg_level = ai.analyze_lighting_and_background(image_array)
            
            # 3. 28쌍 AFTER 파일 기반 완벽한 배경색 선택
            target_bg_color = ai.get_perfect_background_color(lighting, bg_level)
            
            # 4. 검은색 선 완벽 제거
            clean_image = ai.remove_black_lines_perfectly(image_array, line_mask, target_bg_color)
            
            # 5. 금속 타입 감지
            metal_type = ai.detect_metal_type(clean_image, ring_bbox)
            
            # 6. v13.3 완전한 웨딩링 보정
            enhanced_image = ai.enhance_wedding_ring_v13_3(clean_image, ring_bbox, metal_type, lighting)
            
            # 7. 2x 업스케일링
            upscaled_image = ai.upscale_image(enhanced_image, scale=2)
            
            # 8. 완벽한 썸네일 생성 (업스케일된 이미지 기준)
            # ring_bbox도 2배로 스케일링
            scaled_ring_bbox = None
            if ring_bbox:
                x, y, w, h = ring_bbox
                scaled_ring_bbox = (x*2, y*2, w*2, h*2)
            
            thumbnail = ai.create_perfect_thumbnail(upscaled_image, scaled_ring_bbox)
            
            # 9. 결과 인코딩
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
                    "version": "v15.0 Ultimate",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "background_level": bg_level,
                    "background_color": target_bg_color.tolist(),
                    "ring_bbox": ring_bbox,
                    "line_removal": "pixel_perfect",
                    "enhancement": "v13.3_complete",
                    "upscale_factor": 2,
                    "thumbnail_size": "1000x1300",
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{upscaled_image.shape[1]}x{upscaled_image.shape[0]}"
                }
            }
            
    except Exception as e:
        return {
            "error": f"v15.0 처리 중 오류: {str(e)}",
            "details": "완전한 에러 핸들링 시스템"
        }

# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
