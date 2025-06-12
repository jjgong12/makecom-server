from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import time
from datetime import datetime
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class SegmentedWeddingRingEnhancer:
    def __init__(self):
        # 28쌍 학습 데이터 기반 웨딩링 전용 집중 보정
        self.ring_focused_params = {
            'white_gold': {
                'natural': {'brightness': 1.35, 'contrast': 1.25, 'warmth': 0.92, 'saturation': 0.98, 'sharpness': 1.50, 'clarity': 1.30, 'gamma': 1.02},
                'warm': {'brightness': 1.40, 'contrast': 1.30, 'warmth': 0.75, 'saturation': 0.93, 'sharpness': 1.55, 'clarity': 1.35, 'gamma': 1.04},
                'cool': {'brightness': 1.30, 'contrast': 1.20, 'warmth': 0.98, 'saturation': 1.01, 'sharpness': 1.45, 'clarity': 1.25, 'gamma': 1.00}
            },
            'rose_gold': {
                'natural': {'brightness': 1.25, 'contrast': 1.18, 'warmth': 1.25, 'saturation': 1.20, 'sharpness': 1.35, 'clarity': 1.20, 'gamma': 0.99},
                'warm': {'brightness': 1.20, 'contrast': 1.15, 'warmth': 1.08, 'saturation': 1.15, 'sharpness': 1.30, 'clarity': 1.15, 'gamma': 0.96},
                'cool': {'brightness': 1.35, 'contrast': 1.25, 'warmth': 1.40, 'saturation': 1.30, 'sharpness': 1.40, 'clarity': 1.25, 'gamma': 1.03}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.28, 'contrast': 1.22, 'warmth': 1.12, 'saturation': 1.12, 'sharpness': 1.40, 'clarity': 1.25, 'gamma': 1.01},
                'warm': {'brightness': 1.25, 'contrast': 1.20, 'warmth': 0.98, 'saturation': 1.08, 'sharpness': 1.35, 'clarity': 1.22, 'gamma': 0.99},
                'cool': {'brightness': 1.32, 'contrast': 1.25, 'warmth': 1.22, 'saturation': 1.16, 'sharpness': 1.45, 'clarity': 1.28, 'gamma': 1.03}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.30, 'contrast': 1.25, 'warmth': 1.30, 'saturation': 1.25, 'sharpness': 1.38, 'clarity': 1.22, 'gamma': 1.02},
                'warm': {'brightness': 1.22, 'contrast': 1.18, 'warmth': 1.15, 'saturation': 1.17, 'sharpness': 1.33, 'clarity': 1.18, 'gamma': 0.98},
                'cool': {'brightness': 1.38, 'contrast': 1.30, 'warmth': 1.45, 'saturation': 1.33, 'sharpness': 1.45, 'clarity': 1.28, 'gamma': 1.04}
            }
        }
        
        # 28쌍 학습 데이터 기반 배경 전용 분위기 조성
        self.background_focused_params = {
            'natural': {'brightness': 1.08, 'contrast': 1.05, 'warmth': 1.02, 'saturation': 1.03, 'gamma': 0.99},
            'warm': {'brightness': 1.05, 'contrast': 1.03, 'warmth': 0.85, 'saturation': 1.00, 'gamma': 0.97},
            'cool': {'brightness': 1.10, 'contrast': 1.08, 'warmth': 1.15, 'saturation': 1.05, 'gamma': 1.01}
        }
    
    def detect_ring_type(self, image):
        """28쌍 학습 데이터 기반 금속 타입 자동 감지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 금속 영역 추출 (밝기 기반)
        metal_mask = cv2.inRange(hsv[:,:,2], 100, 255)
        metal_pixels = hsv[metal_mask > 0]
        
        if len(metal_pixels) == 0:
            return 'white_gold'
        
        # 색상(Hue) 분석
        avg_hue = np.mean(metal_pixels[:, 0])
        avg_saturation = np.mean(metal_pixels[:, 1])
        
        # 28쌍 분석 결과 기반 임계값
        if avg_saturation < 30:
            return 'white_gold'
        elif 5 <= avg_hue <= 25 and avg_saturation > 40:
            if avg_saturation > 80:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        elif 160 <= avg_hue <= 180 and avg_saturation > 30:
            return 'rose_gold'
        else:
            return 'white_gold'
    
    def detect_lighting_environment(self, image):
        """조명 환경 자동 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128
        
        avg_a = np.mean(a_channel)
        avg_b = np.mean(b_channel)
        
        if avg_b > 8:
            return 'warm'
        elif avg_b < -5:
            return 'cool'
        else:
            return 'natural'
    
    def extract_ring_mask(self, image):
        """웨딩링 영역 자동 추출 (28쌍 학습 패턴 기반)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1단계: 금속 반사 영역 감지 (높은 밝기)
        brightness_mask = cv2.inRange(hsv[:,:,2], 120, 255)
        
        # 2단계: 색상 범위로 금속 감지
        low_saturation = cv2.inRange(hsv[:,:,1], 0, 60)  # 무채색 금속
        
        gold_hue_1 = cv2.inRange(hsv[:,:,0], 10, 30)  # 황색 범위
        gold_hue_2 = cv2.inRange(hsv[:,:,0], 0, 10)   # 주황-황색
        gold_mask = cv2.bitwise_or(gold_hue_1, gold_hue_2)
        
        # 전체 금속 마스크
        metal_mask = cv2.bitwise_and(brightness_mask, 
                                   cv2.bitwise_or(low_saturation, gold_mask))
        
        # 3단계: 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
        
        # 4단계: 컨투어 감지로 링 모양 필터링
        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ring_mask = np.zeros_like(metal_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 300 and area < 80000:
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        cv2.fillPoly(ring_mask, [contour], 255)
        
        # 5단계: 마스크 부드럽게 처리
        ring_mask = cv2.GaussianBlur(ring_mask, (5,5), 2)
        
        return ring_mask
    
    def _prepare_image(self, image):
        """이미지 전처리 및 메모리 최적화"""
        height, width = image.shape[:2]
        
        if width > 2048 or height > 2048:
            scale = min(2048/width, 2048/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def enhance_ring_region(self, image, mask, ring_type, lighting_env):
        """웨딩링 영역 전용 집중 보정"""
        params = self.ring_focused_params[ring_type][lighting_env]
        
        # 마스크 영역만 추출
        ring_region = cv2.bitwise_and(image, image, mask=mask)
        
        # 강력한 금속 보정 적용
        enhanced = self._apply_ring_enhancement(ring_region, params)
        
        return enhanced
    
    def enhance_background_region(self, image, mask, lighting_env):
        """배경 영역 전용 분위기 보정"""
        background_mask = cv2.bitwise_not(mask)
        background_region = cv2.bitwise_and(image, image, mask=background_mask)
        
        params = self.background_focused_params[lighting_env]
        enhanced = self._apply_background_enhancement(background_region, params)
        
        return enhanced
    
    def _apply_ring_enhancement(self, image, params):
        """웨딩링 전용 집중 보정 (금속 질감 극대화)"""
        # 밝기/대비 조정
        enhanced = cv2.convertScaleAbs(image, 
                                     alpha=params['contrast'], 
                                     beta=(params['brightness']-1)*50)
        
        # 색온도 조정
        if params['warmth'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,2] = lab[:,:,2] * params['warmth']
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 채도 조정
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * params['saturation']
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 언샤프 마스킹으로 선명도 극대화
        blurred = cv2.GaussianBlur(enhanced, (5,5), 1.5)
        sharpness = params['sharpness']
        enhanced = cv2.addWeighted(enhanced, 1 + (sharpness-1), 
                                 blurred, -(sharpness-1), 0)
        
        # CLAHE로 명료도 강화
        if params['clarity'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            clip_limit = 2.0 + (params['clarity'] - 1) * 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 감마 보정
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_background_enhancement(self, image, params):
        """배경 전용 분위기 보정 (은은한 조화)"""
        # 부드러운 밝기 조정
        enhanced = cv2.convertScaleAbs(image,
                                     alpha=params['contrast'],
                                     beta=(params['brightness']-1)*30)
        
        # 색온도 조정
        if params['warmth'] != 1.0:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:,:,2] = lab[:,:,2] * params['warmth']
            lab[:,:,2] = np.clip(lab[:,:,2], 0, 255)
            enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # 채도 조정
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * params['saturation']
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 감마 보정
        if params['gamma'] != 1.0:
            inv_gamma = 1.0 / params['gamma']
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def enhance_wedding_ring_segmented(self, image_data):
        """메인 함수: 28쌍 기반 영역별 차별 보정"""
        start_time = time.time()
        
        try:
            # Base64 디코딩
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image.convert('RGB'))
            
            # 이미지 전처리
            image_array = self._prepare_image(image_array)
            
            # AI 분석
            ring_type = self.detect_ring_type(image_array)
            lighting_env = self.detect_lighting_environment(image_array)
            
            # 웨딩링 영역 자동 추출
            ring_mask = self.extract_ring_mask(image_array)
            
            # 각 영역별 차별 보정
            enhanced_ring = self.enhance_ring_region(image_array, ring_mask, ring_type, lighting_env)
            enhanced_background = self.enhance_background_region(image_array, ring_mask, lighting_env)
            
            # 두 영역 자연스럽게 합성
            ring_mask_norm = ring_mask.astype(np.float32) / 255.0
            ring_mask_3d = np.dstack([ring_mask_norm] * 3)
            
            final_result = (enhanced_ring.astype(np.float32) * ring_mask_3d + 
                           enhanced_background.astype(np.float32) * (1 - ring_mask_3d))
            
            # PIL로 변환 및 JPG 저장
            enhanced_pil = Image.fromarray(final_result.astype(np.uint8))
            enhanced_buffer = io.BytesIO()
            enhanced_pil.save(enhanced_buffer, format='JPEG', quality=95, progressive=True)
            enhanced_buffer.seek(0)
            
            processing_time = time.time() - start_time
            
            # 사용된 파라미터
            ring_params = self.ring_focused_params[ring_type][lighting_env]
            bg_params = self.background_focused_params[lighting_env]
            
            logging.info(f"Segmented enhancement: {ring_type} ring under {lighting_env} lighting in {processing_time:.2f}s")
            
            return enhanced_buffer, ring_type, lighting_env, ring_params, bg_params, processing_time
            
        except Exception as e:
            logging.error(f"Segmented enhancement failed: {str(e)}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'segmented_wedding_ring_enhancer'})

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """웨딩링 영역별 차별 보정 - 바이너리 직접 반환"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        enhancer = SegmentedWeddingRingEnhancer()
        enhanced_buffer, ring_type, lighting_env, ring_params, bg_params, processing_time = enhancer.enhance_wedding_ring_segmented(
            data['image_base64']
        )
        
        # 타임스탬프 기반 영어 파일명
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f'segmented_enhanced_{timestamp}.jpg'
        
        return Response(
            enhanced_buffer.getvalue(),
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'attachment; filename={safe_filename}',
                'Content-Type': 'image/jpeg',
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting_env,
                'X-Processing-Time': str(round(processing_time, 2))
            }
        )
        
    except Exception as e:
        logging.error(f"Error in enhance_wedding_ring_segmented: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
