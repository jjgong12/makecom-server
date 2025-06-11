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

class WeddingRingEnhancer:
    def __init__(self):
        # 28쌍 학습 데이터 반영 - 14k 기준 최적 파라미터
        self.metal_lighting_params = {
            'white_gold': {
                'natural': {'brightness': 1.22, 'contrast': 1.12, 'warmth': 0.95, 'saturation': 1.00, 'sharpness': 1.30, 'clarity': 1.18, 'gamma': 1.01},
                'warm': {'brightness': 1.28, 'contrast': 1.18, 'warmth': 0.80, 'saturation': 0.95, 'sharpness': 1.35, 'clarity': 1.22, 'gamma': 1.03},
                'cool': {'brightness': 1.18, 'contrast': 1.08, 'warmth': 1.00, 'saturation': 1.03, 'sharpness': 1.25, 'clarity': 1.15, 'gamma': 0.99}
            },
            'rose_gold': {
                'natural': {'brightness': 1.15, 'contrast': 1.08, 'warmth': 1.20, 'saturation': 1.15, 'sharpness': 1.15, 'clarity': 1.10, 'gamma': 0.98},
                'warm': {'brightness': 1.10, 'contrast': 1.05, 'warmth': 1.05, 'saturation': 1.10, 'sharpness': 1.10, 'clarity': 1.05, 'gamma': 0.95},
                'cool': {'brightness': 1.25, 'contrast': 1.15, 'warmth': 1.35, 'saturation': 1.25, 'sharpness': 1.25, 'clarity': 1.20, 'gamma': 1.02}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.18, 'contrast': 1.12, 'warmth': 1.08, 'saturation': 1.08, 'sharpness': 1.22, 'clarity': 1.15, 'gamma': 1.00},
                'warm': {'brightness': 1.15, 'contrast': 1.10, 'warmth': 0.95, 'saturation': 1.05, 'sharpness': 1.20, 'clarity': 1.12, 'gamma': 0.98},
                'cool': {'brightness': 1.22, 'contrast': 1.15, 'warmth': 1.18, 'saturation': 1.12, 'sharpness': 1.25, 'clarity': 1.18, 'gamma': 1.02}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.20, 'contrast': 1.15, 'warmth': 1.25, 'saturation': 1.20, 'sharpness': 1.18, 'clarity': 1.12, 'gamma': 1.01},
                'warm': {'brightness': 1.12, 'contrast': 1.08, 'warmth': 1.10, 'saturation': 1.12, 'sharpness': 1.15, 'clarity': 1.08, 'gamma': 0.97},
                'cool': {'brightness': 1.28, 'contrast': 1.20, 'warmth': 1.40, 'saturation': 1.28, 'sharpness': 1.25, 'clarity': 1.18, 'gamma': 1.03}
            }
        }
    
    def detect_ring_type(self, image):
        """28쌍 학습 데이터 기반 금속 타입 자동 감지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 금속 영역 추출 (밝기 기반)
        metal_mask = cv2.inRange(hsv[:,:,2], 100, 255)
        metal_pixels = hsv[metal_mask > 0]
        
        if len(metal_pixels) == 0:
            return 'white_gold'  # 기본값
        
        # 색상(Hue) 분석
        avg_hue = np.mean(metal_pixels[:, 0])
        avg_saturation = np.mean(metal_pixels[:, 1])
        
        # 28쌍 분석 결과 기반 임계값
        if avg_saturation < 30:  # 무채색
            return 'white_gold'
        elif 5 <= avg_hue <= 25 and avg_saturation > 40:  # 주황-황색 + 높은 채도
            if avg_saturation > 80:
                return 'yellow_gold'  # 진한 황금색
            else:
                return 'champagne_gold'  # 연한 황금색
        elif 160 <= avg_hue <= 180 and avg_saturation > 30:  # 핑크톤
            return 'rose_gold'
        else:
            return 'white_gold'  # 기본값
    
    def detect_lighting_environment(self, image):
        """조명 환경 자동 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # A, B 채널 분석 (색온도 판단)
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128
        
        avg_a = np.mean(a_channel)
        avg_b = np.mean(b_channel)
        
        # 색온도 판단
        if avg_b > 8:  # 황색 cast 강함
            return 'warm'
        elif avg_b < -5:  # 청색 cast 강함  
            return 'cool'
        else:
            return 'natural'
    
    def _prepare_image(self, image):
        """이미지 전처리 및 메모리 최적화"""
        height, width = image.shape[:2]
        
        # 고해상도 이미지 최적화 (2K 기준)
        if width > 2048 or height > 2048:
            scale = min(2048/width, 2048/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def _adjust_brightness_contrast(self, image, brightness, contrast):
        """밝기 및 대비 조정"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness-1)*50)
    
    def _adjust_warmth(self, image, warmth):
        """색온도 조정 (웨딩링 전용)"""
        if warmth == 1.0:
            return image
            
        # LAB 색공간에서 정확한 색온도 조정
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # B 채널(blue-yellow) 조정으로 색온도 변경
        lab[:, :, 2] = lab[:, :, 2] * warmth
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def _adjust_saturation(self, image, saturation):
        """채도 조정"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _enhance_sharpness(self, image, sharpness):
        """언샤프 마스킹을 이용한 선명도 향상"""
        if sharpness == 1.0:
            return image
            
        # 가우시안 블러
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # 언샤프 마스킹
        sharpened = cv2.addWeighted(image, 1 + (sharpness - 1), blurred, -(sharpness - 1), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _enhance_clarity(self, image, clarity):
        """CLAHE를 이용한 명료도 향상"""
        if clarity == 1.0:
            return image
            
        # LAB 색공간에서 L 채널에만 CLAHE 적용
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        clip_limit = 2.0 + (clarity - 1) * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _apply_gamma_correction(self, image, gamma):
        """감마 보정"""
        if gamma == 1.0:
            return image
            
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def enhance_wedding_ring(self, image_data):
        """메인 웨딩링 보정 함수"""
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
            
            # 최적 파라미터 선택
            params = self.metal_lighting_params[ring_type][lighting_env]
            
            # 7단계 보정 파이프라인
            enhanced = self._adjust_brightness_contrast(
                image_array, params['brightness'], params['contrast']
            )
            enhanced = self._adjust_warmth(enhanced, params['warmth'])
            enhanced = self._adjust_saturation(enhanced, params['saturation'])
            enhanced = self._enhance_sharpness(enhanced, params['sharpness'])
            enhanced = self._enhance_clarity(enhanced, params['clarity'])
            enhanced = self._apply_gamma_correction(enhanced, params['gamma'])
            
            # PIL로 변환 및 JPG 저장
            enhanced_pil = Image.fromarray(enhanced.astype(np.uint8))
            enhanced_buffer = io.BytesIO()
            enhanced_pil.save(enhanced_buffer, format='JPEG', quality=95, progressive=True)
            enhanced_buffer.seek(0)
            
            processing_time = time.time() - start_time
            
            # 처리 결과 로깅
            logging.info(f"Enhanced {ring_type} ring under {lighting_env} lighting in {processing_time:.2f}s")
            
            return enhanced_buffer, ring_type, lighting_env, params, processing_time
            
        except Exception as e:
            logging.error(f"Enhancement failed: {str(e)}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'wedding_ring_enhancer'})

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """웨딩링 보정 - 바이너리 직접 반환"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        enhancer = WeddingRingEnhancer()
        enhanced_buffer, ring_type, lighting_env, params, processing_time = enhancer.enhance_wedding_ring(
            data['image_base64']
        )
        
        # 타임스탬프 기반 영어 파일명 (한글 완전 제거)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f'enhanced_{timestamp}.jpg'
        
        return Response(
            enhanced_buffer.getvalue(),
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'attachment; filename={safe_filename}',
                'Content-Type': 'image/jpeg'
            }
        )
        
    except Exception as e:
        logging.error(f"Error in enhance_wedding_ring_binary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhance_wedding_ring', methods=['POST'])
def enhance_wedding_ring():
    """웨딩링 보정 - JSON 분석 결과 반환"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image_base64 provided'}), 400
        
        enhancer = WeddingRingEnhancer()
        enhanced_buffer, ring_type, lighting_env, params, processing_time = enhancer.enhance_wedding_ring(
            data['image_base64']
        )
        
        # Base64로 인코딩
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        # 분석 결과 JSON 응답 (한글 제거)
        analysis_result = {
            'ring_type': ring_type,
            'lighting': lighting_env,
            'parameters_used': {
                'brightness': float(params['brightness']),
                'contrast': float(params['contrast']), 
                'warmth': float(params['warmth']),
                'saturation': float(params['saturation']),
                'sharpness': float(params['sharpness']),
                'clarity': float(params['clarity']),
                'gamma': float(params['gamma'])
            },
            'processing_time': round(processing_time, 2),
            'message': f'Enhanced {ring_type} ring under {lighting_env} lighting',
            'enhanced_image': enhanced_base64
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logging.error(f"Error in enhance_wedding_ring: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
