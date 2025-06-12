import os
import io
import base64
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, send_file
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class NaturalWeddingRingEnhancer:
    """v5.1 Natural Wedding Ring Enhancement with White Light Overlay"""
    
    def __init__(self):
        # v5.1 강화된 파라미터 - 3번 수준 달성
        self.metal_params = {
            'white_gold': {
                'natural': {'brightness': 1.35, 'contrast': 1.20, 'warmth': 0.90, 'saturation': 1.05, 'sharpness': 1.30, 'clarity': 1.25, 'gamma': 1.03},
                'warm': {'brightness': 1.40, 'contrast': 1.25, 'warmth': 0.75, 'saturation': 1.00, 'sharpness': 1.35, 'clarity': 1.30, 'gamma': 1.05},
                'cool': {'brightness': 1.30, 'contrast': 1.15, 'warmth': 0.95, 'saturation': 1.08, 'sharpness': 1.25, 'clarity': 1.20, 'gamma': 1.02}
            },
            'rose_gold': {
                'natural': {'brightness': 1.25, 'contrast': 1.15, 'warmth': 1.15, 'saturation': 1.20, 'sharpness': 1.20, 'clarity': 1.18, 'gamma': 1.00},
                'warm': {'brightness': 1.20, 'contrast': 1.10, 'warmth': 1.00, 'saturation': 1.15, 'sharpness': 1.15, 'clarity': 1.12, 'gamma': 0.98},
                'cool': {'brightness': 1.35, 'contrast': 1.20, 'warmth': 1.30, 'saturation': 1.30, 'sharpness': 1.30, 'clarity': 1.25, 'gamma': 1.05}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.30, 'contrast': 1.18, 'warmth': 1.05, 'saturation': 1.12, 'sharpness': 1.25, 'clarity': 1.22, 'gamma': 1.02},
                'warm': {'brightness': 1.25, 'contrast': 1.15, 'warmth': 0.92, 'saturation': 1.08, 'sharpness': 1.22, 'clarity': 1.18, 'gamma': 1.00},
                'cool': {'brightness': 1.35, 'contrast': 1.22, 'warmth': 1.15, 'saturation': 1.18, 'sharpness': 1.30, 'clarity': 1.25, 'gamma': 1.05}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.32, 'contrast': 1.20, 'warmth': 1.20, 'saturation': 1.25, 'sharpness': 1.22, 'clarity': 1.20, 'gamma': 1.03},
                'warm': {'brightness': 1.25, 'contrast': 1.15, 'warmth': 1.05, 'saturation': 1.18, 'sharpness': 1.18, 'clarity': 1.15, 'gamma': 1.00},
                'cool': {'brightness': 1.40, 'contrast': 1.25, 'warmth': 1.35, 'saturation': 1.35, 'sharpness': 1.30, 'clarity': 1.25, 'gamma': 1.08}
            }
        }
    
    def detect_ring_metal(self, image):
        """보수적 금속 감지 (애매하면 champagne_gold)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            if h_mean < 15 and s_mean < 50:
                return 'white_gold'
            elif 5 <= h_mean <= 25 and s_mean > 80:
                return 'rose_gold'
            elif 15 <= h_mean <= 35 and 40 <= s_mean <= 90:
                return 'yellow_gold'
            else:
                return 'champagne_gold'  # 기본값
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지 (애매하면 natural)"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean > 135:
                return 'warm'
            elif b_mean < 125:
                return 'cool'
            else:
                return 'natural'  # 기본값
        except:
            return 'natural'
    
    def create_white_light_overlay(self, image):
        """🔥 핵심 기능: 하얀색 조명 오버레이 효과"""
        try:
            height, width = image.shape[:2]
            
            # 상단 중앙에서 시작하는 radial gradient
            center_x, center_y = width // 2, height // 4  # 상단에서 1/4 지점
            
            # 거리 맵 생성
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(width**2 + height**2)
            
            # 하얀 조명 그라디언트 (중앙이 밝고 가장자리로 갈수록 어두워짐)
            gradient = 1.0 - (distance / max_distance)
            gradient = np.clip(gradient, 0.3, 1.0)  # 최소 30% 밝기 보장
            
            # 3채널로 확장
            overlay = np.dstack([gradient, gradient, gradient])
            overlay = (overlay * 255).astype(np.uint8)
            
            return overlay
        except Exception as e:
            logger.error(f"White light overlay error: {e}")
            return np.ones_like(image) * 255
    
    def enhance_ring_basic(self, image, params):
        """기본 링 보정 + 하얀 조명 효과"""
        try:
            # 1. 노이즈 제거
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 2. 하얀 조명 오버레이 적용
            white_overlay = self.create_white_light_overlay(denoised)
            lit_image = cv2.addWeighted(denoised, 0.82, white_overlay, 0.18, 0)
            
            # 3. PIL로 변환하여 세밀한 조정
            pil_image = Image.fromarray(cv2.cvtColor(lit_image, cv2.COLOR_BGR2RGB))
            
            # 4. 밝기 조정
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 5. 대비 조정
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 6. 선명도 조정
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # 7. 채도 조정 (warmth/saturation 반영)
            rgb_array = np.array(enhanced)
            
            # warmth 조정 (색온도)
            if params['warmth'] != 1.0:
                rgb_array[:, :, 0] = np.clip(rgb_array[:, :, 0] * params['warmth'], 0, 255)  # Red
                rgb_array[:, :, 2] = np.clip(rgb_array[:, :, 2] * (2.0 - params['warmth']), 0, 255)  # Blue
            
            # saturation 조정
            if params['saturation'] != 1.0:
                hsv_temp = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv_temp[:, :, 1] = np.clip(hsv_temp[:, :, 1] * params['saturation'], 0, 255)
                rgb_array = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2RGB)
            
            # 8. 감마 보정
            if params['gamma'] != 1.0:
                rgb_array = np.power(rgb_array / 255.0, 1.0 / params['gamma']) * 255.0
                rgb_array = np.clip(rgb_array, 0, 255)
            
            # 9. CLAHE 적용 (clarity)
            if params['clarity'] > 1.0:
                lab = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=params['clarity'], tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                rgb_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 10. 최종 하이라이트 부스팅 (금속 반사 살리기)
            rgb_array = self.boost_highlights(rgb_array)
            
            return cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Basic ring enhancement error: {e}")
            return image
    
    def boost_highlights(self, image):
        """금속 하이라이트 부스팅"""
        try:
            # 밝은 영역 감지 (상위 20%)
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            threshold = np.percentile(gray, 80)
            highlight_mask = gray > threshold
            
            # 하이라이트 영역만 15% 추가 밝기
            boosted = image.copy()
            boosted[highlight_mask] = np.clip(boosted[highlight_mask] * 1.15, 0, 255)
            
            return boosted
        except:
            return image
    
    def _prepare_image(self, image):
        """이미지 전처리 (메모리 최적화)"""
        try:
            height, width = image.shape[:2]
            if max(height, width) > 2048:
                scale = 2048 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return image
        except Exception as e:
            logger.error(f"Image preparation error: {e}")
            return image
    
    def enhance(self, image):
        """메인 보정 함수 - v5.1 하얀 조명 시스템"""
        try:
            # 1. 이미지 전처리
            prepared_image = self._prepare_image(image)
            
            # 2. 자동 분석
            metal_type = self.detect_ring_metal(prepared_image)
            lighting_type = self.detect_lighting(prepared_image)
            
            logger.info(f"Detected: {metal_type} ring, {lighting_type} lighting")
            
            # 3. 파라미터 선택
            params = self.metal_params[metal_type][lighting_type]
            
            # 4. 보정 적용
            enhanced_image = self.enhance_ring_basic(prepared_image, params)
            
            # 5. 원본과 블렌딩 (85:15 - 더 보정된 느낌)
            final_image = cv2.addWeighted(prepared_image, 0.15, enhanced_image, 0.85, 0)
            
            return final_image, metal_type, lighting_type, params
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image, "unknown", "unknown", {}

# 글로벌 enhancer 인스턴스
enhancer = NaturalWeddingRingEnhancer()

@app.route('/')
def home():
    """서버 상태 및 엔드포인트 정보"""
    return jsonify({
        "status": "Wedding Ring AI v5.1 - White Light Overlay System",
        "version": "5.1",
        "features": [
            "Natural 28-pair data based enhancement",
            "White light overlay effect",
            "Professional studio lighting simulation",
            "4 metal types auto-detection",
            "3 lighting conditions adaptation",
            "Highlight boosting for metal reflection"
        ],
        "endpoints": {
            "/health": "Server health check",
            "/enhance_wedding_ring_advanced": "🔥 Main endpoint - White light overlay system",
            "/enhance_wedding_ring_natural": "Natural enhancement",
            "/enhance_wedding_ring_segmented": "Legacy segmented enhancement",
            "/enhance_wedding_ring_binary": "Basic binary enhancement"
        }
    })

@app.route('/health')
def health():
    """서버 상태 확인"""
    return jsonify({"status": "healthy", "version": "5.1"})

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """🔥 메인 엔드포인트 - v5.1 하얀 조명 시스템"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # v5.1 하얀 조명 보정 적용
        enhanced_image, metal_type, lighting_type, params = enhancer.enhance(image)
        
        # JPEG 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        success, encoded_image = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        if not success:
            return jsonify({"error": "Failed to encode enhanced image"}), 500
        
        # 바이너리 데이터로 직접 반환
        enhanced_bytes = io.BytesIO(encoded_image.tobytes())
        enhanced_bytes.seek(0)
        
        logger.info(f"✅ Enhanced: {metal_type} ring, {lighting_type} lighting")
        
        return send_file(
            enhanced_bytes,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='enhanced_wedding_ring_v51.jpg'
        )
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """자연스러운 보정 (v5.1과 동일)"""
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    """기존 세그먼트 보정 (호환성)"""
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    """기본 바이너리 보정 (호환성)"""
    return enhance_wedding_ring_advanced()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
