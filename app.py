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
            'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.08,
            'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
            'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95
        },
        'cool': {
            'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.04,
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
            'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.15,
            'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
            'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98
        },
        'cool': {
            'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.10,
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
            'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.07,
            'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
            'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.97
        },
        'cool': {
            'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.03,
            'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 3,
            'original_blend': 0.15, 'saturation': 1.28, 'gamma': 1.03
        }
    }
}

# 28쌍 AFTER 배경색 (대화 28번 성과)
AFTER_BACKGROUND_COLORS = {
    'natural': [250, 248, 245],
    'warm': [252, 250, 245],
    'cool': [248, 250, 252]
}

class UltimateWeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
        self.after_bg_colors = AFTER_BACKGROUND_COLORS
    
    def detect_actual_line_thickness(self, mask, bbox):
        """대화 29번: 실제 검은색 선 두께 측정 (100픽셀까지 대응)"""
        x, y, w, h = bbox
        
        # 4방향에서 실제 두께 측정
        thicknesses = []
        
        # 상단
        for i in range(min(100, h//2)):
            if np.any(mask[y+i, x:x+w] == 0):
                thicknesses.append(i)
                break
        
        # 하단
        for i in range(min(100, h//2)):
            if np.any(mask[y+h-1-i, x:x+w] == 0):
                thicknesses.append(i)
                break
        
        # 좌측
        for i in range(min(100, w//2)):
            if np.any(mask[y:y+h, x+i] == 0):
                thicknesses.append(i)
                break
        
        # 우측
        for i in range(min(100, w//2)):
            if np.any(mask[y:y+h, x+w-1-i] == 0):
                thicknesses.append(i)
                break
        
        if thicknesses:
            # 중간값 사용 + 50% 오차 범위
            actual = int(np.median(thicknesses))
            return int(actual * 1.5)
        return 50
    
    def detect_black_border_edges_only(self, image):
        """대화 34번: 가장자리 50픽셀에서만 검은색 선 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # 가장자리 마스크 생성
        edge_mask = np.zeros_like(gray)
        edge_width = 50
        
        # 가장자리만 활성화
        edge_mask[:edge_width, :] = 255  # 상단
        edge_mask[-edge_width:, :] = 255  # 하단
        edge_mask[:, :edge_width] = 255  # 좌측
        edge_mask[:, -edge_width:] = 255  # 우측
        
        # 가장자리에서만 검은색 찾기
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        border_only = cv2.bitwise_and(binary, edge_mask)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(border_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # 전체 테두리 마스크 생성
            full_mask = np.zeros_like(gray)
            cv2.drawContours(full_mask, [largest], -1, 255, -1)
            
            return full_mask, (x, y, w, h)
        
        return None, None
    
    def remove_border_direct_replacement(self, image, mask, bbox, lighting='natural'):
        """대화 27-28번: 배경색 직접 교체 방식"""
        if mask is None:
            return image
        
        result = image.copy()
        x, y, w, h = bbox
        
        # 실제 두께 측정
        thickness = self.detect_actual_line_thickness(mask, bbox)
        thickness = min(thickness, 100)  # 최대 100픽셀
        
        # AFTER 배경색 가져오기
        bg_color = np.array(self.after_bg_colors[lighting])
        
        # 가장자리 30픽셀만 제거 (웨딩링 보호)
        edge_only_mask = np.zeros_like(mask)
        edge_width = 30
        
        # 상하좌우 가장자리만
        edge_only_mask[y:y+edge_width, x:x+w] = 255  # 상단
        edge_only_mask[y+h-edge_width:y+h, x:x+w] = 255  # 하단
        edge_only_mask[y:y+h, x:x+edge_width] = 255  # 좌측
        edge_only_mask[y:y+h, x+w-edge_width:x+w] = 255  # 우측
        
        # 웨딩링 보호 영역 제외
        inner_margin = thickness + 50
        inner_x = x + inner_margin
        inner_y = y + inner_margin
        inner_w = w - 2 * inner_margin
        inner_h = h - 2 * inner_margin
        
        if inner_w > 100 and inner_h > 100:
            edge_only_mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w] = 0
        
        # 배경색으로 직접 교체
        mask_indices = np.where(edge_only_mask > 0)
        result[mask_indices] = bg_color
        
        # 부드러운 블렌딩
        blurred_mask = cv2.GaussianBlur(edge_only_mask.astype(np.float32), (31, 31), 10)
        blurred_mask = blurred_mask / 255.0
        
        for c in range(3):
            result[:,:,c] = (image[:,:,c].astype(np.float32) * (1 - blurred_mask) + 
                           result[:,:,c].astype(np.float32) * blurred_mask)
        
        return result.astype(np.uint8), (inner_x, inner_y, inner_w, inner_h)
    
    def detect_metal_and_lighting(self, image):
        """금속 타입과 조명 감지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 중앙 영역에서 분석
        h, w = image.shape[:2]
        center_y, center_x = h//2, w//2
        roi_size = min(h, w) // 4
        
        roi_hsv = hsv[center_y-roi_size:center_y+roi_size, 
                      center_x-roi_size:center_x+roi_size]
        roi_lab = lab[center_y-roi_size:center_y+roi_size,
                      center_x-roi_size:center_x+roi_size]
        
        avg_hue = np.mean(roi_hsv[:,:,0])
        avg_sat = np.mean(roi_hsv[:,:,1])
        b_mean = np.mean(roi_lab[:,:,2])
        
        # 금속 타입
        if avg_sat < 30:
            metal = 'white_gold'
        elif 15 <= avg_hue <= 25:
            metal = 'champagne_gold'
        elif avg_hue < 15 or avg_hue > 165:
            metal = 'rose_gold'
        else:
            metal = 'yellow_gold'
        
        # 조명
        if b_mean < 125:
            lighting = 'warm'
        elif b_mean > 135:
            lighting = 'cool'
        else:
            lighting = 'natural'
        
        return metal, lighting
    
    def apply_v13_enhancement(self, image, params):
        """v13.3 완전한 10단계 보정"""
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # PIL로 변환
        pil_image = Image.fromarray(denoised)
        
        # 2. 밝기
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 3. 대비
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 4. 선명도
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # 5. 채도
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(params.get('saturation', 1.1))
        
        # numpy 배열로
        enhanced_array = np.array(enhanced)
        
        # 6. 하얀색 오버레이
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 7. LAB 색온도 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab = lab.astype(np.float32)
        lab[:,:,1] = np.clip(lab[:,:,1] + params['color_temp_a'], 0, 255)
        lab[:,:,2] = np.clip(lab[:,:,2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 8. CLAHE
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 9. 감마 보정
        gamma = params.get('gamma', 1.0)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced_array = cv2.LUT(enhanced_array, table)
        
        # 10. 원본과 블렌딩
        final = cv2.addWeighted(
            enhanced_array, 1 - params['original_blend'],
            image, params['original_blend'], 0
        )
        
        return final
    
    def create_perfect_thumbnail(self, image, bbox=None):
        """완벽한 1000x1300 썸네일"""
        h, w = image.shape[:2]
        
        if bbox:
            x, y, bw, bh = bbox
            # 웨딩링이 80% 차지하도록
            margin = 0.1
            crop_x = max(0, int(x - bw * margin))
            crop_y = max(0, int(y - bh * margin))
            crop_w = min(w - crop_x, int(bw * (1 + 2 * margin)))
            crop_h = min(h - crop_y, int(bh * (1 + 2 * margin)))
        else:
            # 중앙 영역 크롭
            crop_size = min(h, w) * 0.8
            crop_x = int((w - crop_size) / 2)
            crop_y = int((h - crop_size) / 2)
            crop_w = crop_h = int(crop_size)
        
        cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # 1000x1300 리사이즈
        target_w, target_h = 1000, 1300
        ratio = min(target_w / crop_w, target_h / crop_h) * 0.9
        
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 캔버스에 배치
        canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
        start_y = (target_h - new_h) // 4  # 위쪽으로
        start_x = (target_w - new_w) // 2
        
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas

def handler(event):
    """메인 핸들러"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(event["input"]["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        processor = UltimateWeddingRingProcessor()
        
        # 1. 금속/조명 감지
        metal_type, lighting = processor.detect_metal_and_lighting(image_array)
        params = processor.params.get(metal_type, {}).get(lighting, 
                                     processor.params['white_gold']['natural'])
        
        # 2. v13.3 보정 (무조건 실행)
        enhanced = processor.apply_v13_enhancement(image_array, params)
        
        # 3. 검은색 테두리 감지 (가장자리만)
        border_mask, border_bbox = processor.detect_black_border_edges_only(enhanced)
        
        # 4. 테두리 제거
        if border_mask is not None:
            final_image, ring_bbox = processor.remove_border_direct_replacement(
                enhanced, border_mask, border_bbox, lighting
            )
        else:
            final_image = enhanced
            ring_bbox = None
        
        # 5. 2x 업스케일링
        height, width = final_image.shape[:2]
        upscaled = cv2.resize(final_image, (width*2, height*2), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # 6. 썸네일 생성
        thumbnail = processor.create_perfect_thumbnail(upscaled, ring_bbox)
        
        # 7. 결과 인코딩
        # 메인 이미지
        main_pil = Image.fromarray(upscaled)
        main_buffer = io.BytesIO()
        main_pil.save(main_buffer, format='JPEG', quality=95)
        main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
        
        # 썸네일
        thumb_pil = Image.fromarray(thumbnail)
        thumb_buffer = io.BytesIO()
        thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
        thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
        
        return {
            "enhanced_image": main_base64,
            "thumbnail": thumb_base64,
            "metal_type": metal_type,
            "lighting": lighting
        }
        
    except Exception as e:
        # 에러여도 실제 처리 시도
        try:
            # 최소한 기본 보정이라도
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.2, beta=10)
            upscaled = cv2.resize(enhanced, (width*2, height*2), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            main_pil = Image.fromarray(upscaled)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            # 썸네일도 생성
            thumbnail = cv2.resize(enhanced, (1000, 1300), 
                                 interpolation=cv2.INTER_LANCZOS4)
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "error": str(e)
            }
        except:
            return {"error": f"완전 실패: {str(e)}"}

runpod.serverless.start({"handler": handler})
