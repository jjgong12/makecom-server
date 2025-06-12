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

class SafeBrightnessWeddingRingEnhancer:
    """v5.4 Safe Brightness Wedding Ring Enhancement"""
    
    def __init__(self):
        # v5.4 안전한 파라미터 - 28쌍 베이스 + 단순 밝기 강화만
        self.metal_params = {
            'white_gold': {
                'natural': {'brightness': 1.28, 'contrast': 1.18, 'sharpness': 1.35, 'clarity': 1.20},
                'warm': {'brightness': 1.32, 'contrast': 1.22, 'sharpness': 1.40, 'clarity': 1.25},
                'cool': {'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.30, 'clarity': 1.18}
            },
            'rose_gold': {
                'natural': {'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.25, 'clarity': 1.18},
                'warm': {'brightness': 1.22, 'contrast': 1.12, 'sharpness': 1.20, 'clarity': 1.15},
                'cool': {'brightness': 1.30, 'contrast': 1.20, 'sharpness': 1.30, 'clarity': 1.22}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.27, 'contrast': 1.17, 'sharpness': 1.28, 'clarity': 1.20},
                'warm': {'brightness': 1.24, 'contrast': 1.14, 'sharpness': 1.25, 'clarity': 1.18},
                'cool': {'brightness': 1.30, 'contrast': 1.20, 'sharpness': 1.32, 'clarity': 1.22}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.26, 'contrast': 1.16, 'sharpness': 1.26, 'clarity': 1.19},
                'warm': {'brightness': 1.23, 'contrast': 1.13, 'sharpness': 1.22, 'clarity': 1.16},
                'cool': {'brightness': 1.32, 'contrast': 1.22, 'sharpness': 1.32, 'clarity': 1.25}
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
    
    def enhance_ring_safe_brightness(self, image, params):
        """🔥 완전 안전한 밝기 강화 - 단순하고 확실하게"""
        try:
            # 1. 노이즈 제거 (기본)
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 2. PIL로 변환 (가장 안전한 방식)
            pil_image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            # 3. 밝기 강화 (28쌍 베이스 + 15% UP)
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 4. 대비 강화 (더 깨끗하고 선명하게)
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 5. 선명도 강화 (웨딩링 디테일)
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # 6. numpy 배열로 변환
            rgb_array = np.array(enhanced)
            
            # 7. 명료도 향상 (CLAHE - 안전하게)
            if params['clarity'] > 1.0:
                lab = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
                # 안전한 강도로 제한
                safe_clarity = min(params['clarity'], 1.3)
                clahe = cv2.createCLAHE(clipLimit=safe_clarity, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                rgb_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 8. 미묘한 하이라이트 부스팅 (10%만)
            rgb_array = self.safe_highlight_boost(rgb_array)
            
            return cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Safe brightness enhancement error: {e}")
            return image
    
    def safe_highlight_boost(self, image):
        """안전한 하이라이트 부스팅 (최소한으로)"""
        try:
            # 매우 밝은 영역만 (상위 15%)
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            threshold = np.percentile(gray, 85)
            highlight_mask = gray > threshold
            
            # 매우 미묘하게 8% 부스팅
            boosted = image.copy()
            boosted[highlight_mask] = np.clip(boosted[highlight_mask] * 1.08, 0, 255)
            
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
        """메인 보정 함수 - v5.4 완전 안전한 밝기 강화"""
        try:
            # 1. 이미지 전처리
            prepared_image = self._prepare_image(image)
            
            # 2. 자동 분석
            metal_type = self.detect_ring_metal(prepared_image)
            lighting_type = self.detect_lighting(prepared_image)
            
            logger.info(f"Detected: {metal_type} ring, {lighting_type} lighting")
            
            # 3. 안전한 파라미터 선택 (4개 항목만)
            params = self.metal_params[metal_type][lighting_type]
            
            # 4. 안전한 밝기 강화 적용
            enhanced_image = self.enhance_ring_safe_brightness(prepared_image, params)
            
            # 5. 원본과 자연스럽게 블렌딩 (80:20 - 안전하게)
            final_image = cv2.addWeighted(prepared_image, 0.20, enhanced_image, 0.80, 0)
            
            return final_image, metal_type, lighting_type, params
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image, "unknown", "unknown", {}

# 글로벌 enhancer 인스턴스
enhancer = SafeBrightnessWeddingRingEnhancer()

@app.route('/')
def home():
    """서버 상태 및 엔드포인트 정보"""
    return jsonify({
        "status": "Wedding Ring AI v5.4 - Safe Brightness Enhancement",
        "version": "5.4",
        "features": [
            "28-pair data based safe enhancement",
            "Simple brightness and contrast boost only",
            "No color temperature changes",
            "No saturation modifications", 
            "PIL-based safe processing",
            "Minimal highlight boosting"
        ],
        "safety": [
            "No warmth adjustments (background safe)",
            "No saturation changes (color safe)",
            "Limited clarity enhancement",
            "Conservative blending ratio"
        ],
        "endpoints": {
            "/health": "Server health check",
            "/enhance_wedding_ring_advanced": "🔥 Main endpoint - Safe brightness system",
            "/enhance_wedding_ring_natural": "Natural enhancement",
            "/enhance_wedding_ring_segmented": "Legacy segmented enhancement",
            "/enhance_wedding_ring_binary": "Basic binary enhancement"
        }
    })

@app.route('/health')
def health():
    """서버 상태 확인"""
    return jsonify({"status": "healthy", "version": "5.4"})

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """🔥 메인 엔드포인트 - v5.4 완전 안전한 밝기 강화 시스템"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # v5.4 완전 안전한 밝기 강화 적용
        enhanced_image, metal_type, lighting_type, params = enhancer.enhance(image)
        
        # JPEG 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        success, encoded_image = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        if not success:
            return jsonify({"error": "Failed to encode enhanced image"}), 500
        
        # 바이너리 데이터로 직접 반환
        enhanced_bytes = io.BytesIO(encoded_image.tobytes())
        enhanced_bytes.seek(0)
        
        logger.info(f"✅ Enhanced (Safe Brightness): {metal_type} ring, {lighting_type} lighting")
        
        return send_file(
            enhanced_bytes,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='enhanced_wedding_ring_v54.jpg'
        )
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """자연스러운 보정 (v5.4와 동일)"""
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
