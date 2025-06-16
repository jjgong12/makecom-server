import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io

# v13.3 완전한 파라미터 - 28쌍 학습 데이터 기반
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {'brightness': 1.18, 'contrast': 1.12, 'white_overlay': 0.09,
                   'sharpness': 1.15, 'color_temp_a': -3, 'color_temp_b': -3,
                   'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.01},
        'warm': {'brightness': 1.16, 'contrast': 1.10, 'white_overlay': 0.12,
                'sharpness': 1.13, 'color_temp_a': -5, 'color_temp_b': -5,
                'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98},
        'cool': {'brightness': 1.20, 'contrast': 1.14, 'white_overlay': 0.07,
                'sharpness': 1.17, 'color_temp_a': -2, 'color_temp_b': -2,
                'original_blend': 0.12, 'saturation': 1.03, 'gamma': 1.02}
    },
    'rose_gold': {
        'natural': {'brightness': 1.15, 'contrast': 1.08, 'white_overlay': 0.06,
                   'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
                   'original_blend': 0.20, 'saturation': 1.15, 'gamma': 0.98},
        'warm': {'brightness': 1.10, 'contrast': 1.05, 'white_overlay': 0.08,
                'sharpness': 1.10, 'color_temp_a': 1, 'color_temp_b': 0,
                'original_blend': 0.25, 'saturation': 1.10, 'gamma': 0.95},
        'cool': {'brightness': 1.25, 'contrast': 1.15, 'white_overlay': 0.04,
                'sharpness': 1.25, 'color_temp_a': 3, 'color_temp_b': 2,
                'original_blend': 0.15, 'saturation': 1.25, 'gamma': 1.02}
    },
    'champagne_gold': {
        'natural': {'brightness': 1.17, 'contrast': 1.11, 'white_overlay': 0.12,
                   'sharpness': 1.16, 'color_temp_a': -4, 'color_temp_b': -4,
                   'original_blend': 0.15, 'saturation': 1.02, 'gamma': 1.00},
        'warm': {'brightness': 1.15, 'contrast': 1.10, 'white_overlay': 0.15,
                'sharpness': 1.20, 'color_temp_a': -6, 'color_temp_b': -6,
                'original_blend': 0.18, 'saturation': 1.00, 'gamma': 0.98},
        'cool': {'brightness': 1.22, 'contrast': 1.15, 'white_overlay': 0.10,
                'sharpness': 1.25, 'color_temp_a': -3, 'color_temp_b': -3,
                'original_blend': 0.12, 'saturation': 1.05, 'gamma': 1.02}
    },
    'yellow_gold': {
        'natural': {'brightness': 1.16, 'contrast': 1.09, 'white_overlay': 0.05,
                   'sharpness': 1.14, 'color_temp_a': 3, 'color_temp_b': 2,
                   'original_blend': 0.22, 'saturation': 1.20, 'gamma': 1.01},
        'warm': {'brightness': 1.12, 'contrast': 1.08, 'white_overlay': 0.07,
                'sharpness': 1.15, 'color_temp_a': 2, 'color_temp_b': 1,
                'original_blend': 0.25, 'saturation': 1.12, 'gamma': 0.97},
        'cool': {'brightness': 1.28, 'contrast': 1.20, 'white_overlay': 0.03,
                'sharpness': 1.25, 'color_temp_a': 5, 'color_temp_b': 3,
                'original_blend': 0.15, 'saturation': 1.28, 'gamma': 1.03}
    }
}

class FinalWeddingRingProcessor:
    def __init__(self):
        self.params = WEDDING_RING_PARAMS
    
    def find_black_border(self, image):
        """검은색 테두리 찾기 - 더 확실하게"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # 여러 threshold로 시도
        for thresh_val in [40, 50, 60, 70]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # 가장자리 150픽셀 확인 (100픽셀 두께 + 여유)
            edge_mask = np.zeros_like(binary)
            edge_width = 150
            edge_mask[:edge_width, :] = 255
            edge_mask[-edge_width:, :] = 255
            edge_mask[:, :edge_width] = 255
            edge_mask[:, -edge_width:] = 255
            
            edge_binary = cv2.bitwise_and(binary, edge_mask)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 사각형 모양 컨투어 찾기
                for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                    x, y, cw, ch = cv2.boundingRect(contour)
                    # 전체 이미지 크기의 50% 이상이면 테두리로 간주
                    if cw > w * 0.5 and ch > h * 0.5:
                        return True, (x, y, cw, ch)
        
        return False, None
    
    def remove_black_border_completely(self, image, bbox):
        """검은색 테두리 완전 제거"""
        if bbox is None:
            return image
        
        x, y, w, h = bbox
        result = image.copy()
        
        # 테두리 두께 계산 (상하좌우 중 최소값)
        thickness = min(
            min(100, y),  # 상단
            min(100, x),  # 좌측
            min(100, image.shape[0] - (y + h)),  # 하단
            min(100, image.shape[1] - (x + w))   # 우측
        )
        
        if thickness < 10:
            thickness = 50  # 최소 50픽셀
        
        # 깨끗한 배경색
        bg_color = np.array([250, 248, 245])
        
        # 테두리 영역을 배경색으로 직접 교체
        # 상단
        result[:y+thickness, :] = bg_color
        # 하단
        result[y+h-thickness:, :] = bg_color
        # 좌측
        result[:, :x+thickness] = bg_color
        # 우측
        result[:, x+w-thickness:] = bg_color
        
        # 웨딩링 영역 (테두리 제거 후 남은 부분)
        ring_x = x + thickness
        ring_y = y + thickness
        ring_w = w - 2 * thickness
        ring_h = h - 2 * thickness
        
        # 부드러운 블렌딩을 위한 마스크
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[ring_y:ring_y+ring_h, ring_x:ring_x+ring_w] = 1.0
        
        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        
        # 블렌딩
        for c in range(3):
            result[:,:,c] = (result[:,:,c] * (1 - mask) + image[:,:,c] * mask).astype(np.uint8)
        
        return result, (ring_x, ring_y, ring_w, ring_h)
    
    def apply_v13_enhancement(self, image, metal='champagne_gold', lighting='natural'):
        """v13.3 완전한 보정"""
        params = self.params.get(metal, {}).get(lighting, self.params['champagne_gold']['natural'])
        
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # PIL 변환
        pil_image = Image.fromarray(denoised)
        
        # 2-5. 기본 보정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(params.get('saturation', 1.1))
        
        # numpy 변환
        enhanced_array = np.array(enhanced)
        
        # 6. 하얀색 오버레이
        white = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(enhanced_array, 1 - params['white_overlay'],
                                       white, params['white_overlay'], 0)
        
        # 7. LAB 색온도
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:,:,1] = np.clip(lab[:,:,1] + params['color_temp_a'], 0, 255)
        lab[:,:,2] = np.clip(lab[:,:,2] + params['color_temp_b'], 0, 255)
        enhanced_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 8. CLAHE
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 9. 감마
        gamma = params.get('gamma', 1.0)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced_array = cv2.LUT(enhanced_array, table)
        
        # 10. 원본 블렌딩
        final = cv2.addWeighted(enhanced_array, 1 - params['original_blend'],
                              image, params['original_blend'], 0)
        
        return final
    
    def create_thumbnail_smart(self, image, ring_bbox=None):
        """스마트 썸네일 생성"""
        h, w = image.shape[:2]
        
        if ring_bbox and ring_bbox[2] > 100 and ring_bbox[3] > 100:
            # 웨딩링 영역이 있으면 그 기준으로
            x, y, rw, rh = ring_bbox
            # 여백 추가
            margin = 0.2
            crop_x = max(0, int(x - rw * margin))
            crop_y = max(0, int(y - rh * margin))
            crop_w = min(w - crop_x, int(rw * (1 + 2 * margin)))
            crop_h = min(h - crop_y, int(rh * (1 + 2 * margin)))
        else:
            # 없으면 중앙 기준
            size = int(min(h, w) * 0.6)
            crop_x = (w - size) // 2
            crop_y = (h - size) // 2
            crop_w = crop_h = size
        
        # 크롭
        cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # 1000x1300으로 리사이즈
        target_w, target_h = 1000, 1300
        
        # 비율 계산 (여백 최소화)
        scale = min(target_w / crop_w, target_h / crop_h) * 0.95
        
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        # 리사이즈
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 캔버스 생성
        canvas = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
        
        # 중앙 배치
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas

def handler(event):
    """메인 핸들러"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(event["input"]["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        processor = FinalWeddingRingProcessor()
        
        # 1. v13.3 보정 먼저 적용
        enhanced = processor.apply_v13_enhancement(image_array)
        
        # 2. 검은색 테두리 찾기
        has_border, border_bbox = processor.find_black_border(enhanced)
        
        # 3. 테두리 제거
        if has_border and border_bbox:
            final_image, ring_bbox = processor.remove_black_border_completely(enhanced, border_bbox)
        else:
            final_image = enhanced
            # 테두리 없으면 전체를 웨딩링 영역으로
            h, w = enhanced.shape[:2]
            ring_bbox = (w//4, h//4, w//2, h//2)
        
        # 4. 추가 보정 (더 밝고 선명하게)
        final_image = cv2.convertScaleAbs(final_image, alpha=1.1, beta=10)
        
        # 5. 2x 업스케일링
        height, width = final_image.shape[:2]
        upscaled = cv2.resize(final_image, (width*2, height*2), interpolation=cv2.INTER_LANCZOS4)
        
        # 6. 썸네일 생성
        thumbnail = processor.create_thumbnail_smart(upscaled, 
                                                   (ring_bbox[0]*2, ring_bbox[1]*2, 
                                                    ring_bbox[2]*2, ring_bbox[3]*2))
        
        # 7. 인코딩
        # 메인
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
            "thumbnail": thumb_base64
        }
        
    except Exception as e:
        # 에러여도 무조건 결과 반환
        # 최소한 밝게라도
        try:
            enhanced = cv2.convertScaleAbs(image_array, alpha=1.3, beta=20)
            upscaled = cv2.resize(enhanced, (image_array.shape[1]*2, image_array.shape[0]*2))
            
            # 중앙 크롭 썸네일
            h, w = upscaled.shape[:2]
            size = min(h, w) // 2
            cx, cy = w//2, h//2
            cropped = upscaled[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
            thumbnail = cv2.resize(cropped, (1000, 1300))
            
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
                "error": str(e)
            }
        except:
            # 그래도 실패하면 원본이라도
            return {
                "enhanced_image": base64.b64encode(image_data).decode(),
                "thumbnail": base64.b64encode(image_data).decode(),
                "error": f"완전실패: {str(e)}"
            }

runpod.serverless.start({"handler": handler})
