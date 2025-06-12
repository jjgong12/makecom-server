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

class LightroomStyleWeddingRingEnhancer:
    """v5.2 Lightroom Style Wedding Ring Enhancement"""
    
    def __init__(self):
        # v5.2 라이트룸 스타일 파라미터
        self.metal_params = {
            'white_gold': {
                'natural': {'exposure': 0.6, 'highlights': -25, 'whites': 35, 'shadows': 15, 'contrast': 15, 'clarity': 25, 'vibrance': 8, 'warmth': -300},
                'warm': {'exposure': 0.7, 'highlights': -30, 'whites': 40, 'shadows': 20, 'contrast': 18, 'clarity': 30, 'vibrance': 5, 'warmth': -500},
                'cool': {'exposure': 0.5, 'highlights': -20, 'whites': 30, 'shadows': 12, 'contrast': 12, 'clarity': 20, 'vibrance': 10, 'warmth': -200}
            },
            'rose_gold': {
                'natural': {'exposure': 0.5, 'highlights': -20, 'whites': 30, 'shadows': 18, 'contrast': 12, 'clarity': 20, 'vibrance': 15, 'warmth': 200},
                'warm': {'exposure': 0.4, 'highlights': -15, 'whites': 25, 'shadows': 15, 'contrast': 10, 'clarity': 18, 'vibrance': 12, 'warmth': 100},
                'cool': {'exposure': 0.7, 'highlights': -25, 'whites': 35, 'shadows': 22, 'contrast': 15, 'clarity': 25, 'vibrance': 20, 'warmth': 400}
            },
            'champagne_gold': {
                'natural': {'exposure': 0.6, 'highlights': -22, 'whites': 32, 'shadows': 16, 'contrast': 14, 'clarity': 22, 'vibrance': 12, 'warmth': 150},
                'warm': {'exposure': 0.5, 'highlights': -18, 'whites': 28, 'shadows': 14, 'contrast': 12, 'clarity': 20, 'vibrance': 10, 'warmth': 50},
                'cool': {'exposure': 0.7, 'highlights': -25, 'whites': 35, 'shadows': 18, 'contrast': 16, 'clarity': 25, 'vibrance': 15, 'warmth': 250}
            },
            'yellow_gold': {
                'natural': {'exposure': 0.6, 'highlights': -20, 'whites': 35, 'shadows': 18, 'contrast': 15, 'clarity': 22, 'vibrance': 18, 'warmth': 300},
                'warm': {'exposure': 0.5, 'highlights': -15, 'whites': 30, 'shadows': 15, 'contrast': 12, 'clarity': 20, 'vibrance': 15, 'warmth': 200},
                'cool': {'exposure': 0.8, 'highlights': -30, 'whites': 40, 'shadows': 22, 'contrast': 18, 'clarity': 25, 'vibrance': 25, 'warmth': 500}
            }
        }
    
    def detect_ring_metal(self, image):
        """보수적 금속 감지"""
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
                return 'champagne_gold'
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean > 135:
                return 'warm'
            elif b_mean < 125:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def apply_lightroom_style(self, image, params):
        """🔥 라이트룸 스타일 보정 (균등하고 자연스럽게)"""
        try:
            # 1. 노이즈 제거
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 2. LAB 색공간으로 변환 (라이트룸과 유사한 처리)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # 3. Exposure (전체적인 밝기) - 라이트룸의 Exposure 슬라이더
            exposure_factor = 1.0 + (params['exposure'] / 2.0)  # 0.6 → 1.3배
            l_channel = np.clip(l_channel.astype(np.float32) * exposure_factor, 0, 255)
            
            # 4. Highlights (밝은 부분 조정) - 라이트룸의 Highlights 슬라이더
            if params['highlights'] != 0:
                highlight_mask = l_channel > 180  # 밝은 영역 마스크
                highlight_adjustment = 1.0 + (params['highlights'] / 100.0)  # -25 → 0.75배
                l_channel[highlight_mask] = np.clip(l_channel[highlight_mask] * highlight_adjustment, 0, 255)
            
            # 5. Whites (화이트 포인트) - 라이트룸의 Whites 슬라이더
            if params['whites'] != 0:
                white_mask = l_channel > 200  # 매우 밝은 영역
                white_adjustment = params['whites'] / 100.0  # 35 → +0.35
                l_channel[white_mask] = np.clip(l_channel[white_mask] + white_adjustment * 50, 0, 255)
            
            # 6. Shadows (어두운 부분 들어올리기) - 라이트룸의 Shadows 슬라이더
            if params['shadows'] != 0:
                shadow_mask = l_channel < 100  # 어두운 영역 마스크
                shadow_adjustment = 1.0 + (params['shadows'] / 100.0)  # 15 → 1.15배
                l_channel[shadow_mask] = np.clip(l_channel[shadow_mask] * shadow_adjustment, 0, 255)
            
            # 7. LAB 다시 합성
            lab_enhanced = cv2.merge([l_channel.astype(np.uint8), a_channel, b_channel])
            rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # 8. RGB에서 추가 조정
            rgb_array = rgb_enhanced.astype(np.float32)
            
            # 9. Temperature/Warmth 조정 (라이트룸의 Temp 슬라이더)
            if params['warmth'] != 0:
                temp_factor = params['warmth'] / 1000.0  # -300 → -0.3
                # 음수: 차갑게 (블루 증가), 양수: 따뜻하게 (레드 증가)
                if temp_factor < 0:  # 차갑게
                    rgb_array[:, :, 2] = np.clip(rgb_array[:, :, 2] * (1.0 - temp_factor), 0, 255)  # Blue 증가
                    rgb_array[:, :, 0] = np.clip(rgb_array[:, :, 0] * (1.0 + temp_factor), 0, 255)  # Red 감소
                else:  # 따뜻하게
                    rgb_array[:, :, 0] = np.clip(rgb_array[:, :, 0] * (1.0 + temp_factor), 0, 255)  # Red 증가
                    rgb_array[:, :, 2] = np.clip(rgb_array[:, :, 2] * (1.0 - temp_factor), 0, 255)  # Blue 감소
            
            # 10. Vibrance (자연스러운 채도) - 라이트룸의 Vibrance 슬라이더
            if params['vibrance'] != 0:
                hsv_temp = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_BGR2HSV)
                vibrance_factor = 1.0 + (params['vibrance'] / 100.0)  # 15 → 1.15배
                
                # Vibrance는 이미 포화된 색상은 적게, 덜 포화된 색상은 많이 조정
                saturation_mask = hsv_temp[:, :, 1] < 128  # 덜 포화된 영역만
                hsv_temp[saturation_mask, 1] = np.clip(hsv_temp[saturation_mask, 1] * vibrance_factor, 0, 255)
                
                rgb_array = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR).astype(np.float32)
            
            # 11. Contrast (대비) - 라이트룸의 Contrast 슬라이더
            if params['contrast'] != 0:
                contrast_factor = 1.0 + (params['contrast'] / 100.0)  # 15 → 1.15배
                # 128을 중심으로 대비 조정
                rgb_array = np.clip(128 + (rgb_array - 128) * contrast_factor, 0, 255)
            
            # 12. Clarity (명료도) - 라이트룸의 Clarity 슬라이더
            if params['clarity'] > 0:
                clarity_factor = 1.0 + (params['clarity'] / 100.0)  # 25 → 1.25배
                # 언샤프 마스킹으로 미드톤 대비 향상
                blurred = cv2.GaussianBlur(rgb_array.astype(np.uint8), (0, 0), 2.0)
                mask = rgb_array.astype(np.float32) - blurred.astype(np.float32)
                rgb_array = np.clip(rgb_array + mask * (clarity_factor - 1.0), 0, 255)
            
            return rgb_array.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Lightroom style enhancement error: {e}")
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
        """메인 보정 함수 - v5.2 라이트룸 스타일"""
        try:
            # 1. 이미지 전처리
            prepared_image = self._prepare_image(image)
            
            # 2. 자동 분석
            metal_type = self.detect_ring_metal(prepared_image)
            lighting_type = self.detect_lighting(prepared_image)
            
            logger.info(f"Detected: {metal_type} ring, {lighting_type} lighting")
            
            # 3. 라이트룸 파라미터 선택
            params = self.metal_params[metal_type][lighting_type]
            
            # 4. 라이트룸 스타일 보정 적용
            enhanced_image = self.apply_lightroom_style(prepared_image, params)
            
            # 5. 원본과 자연스럽게 블렌딩 (90:10 - 라이트룸처럼 자연스럽게)
            final_image = cv2.addWeighted(prepared_image, 0.10, enhanced_image, 0.90, 0)
            
            return final_image, metal_type, lighting_type, params
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image, "unknown", "unknown", {}

# 글로벌 enhancer 인스턴스
enhancer = LightroomStyleWeddingRingEnhancer()

@app.route('/')
def home():
    """서버 상태 및 엔드포인트 정보"""
    return jsonify({
        "status": "Wedding Ring AI v5.2 - Lightroom Style Enhancement",
        "version": "5.2",
        "features": [
            "Lightroom-style natural enhancement",
            "Exposure, Highlights, Whites, Shadows adjustment",
            "Temperature and Vibrance control",
            "Clarity and Contrast optimization",
            "Professional photo editing simulation",
            "Uniform brightness enhancement"
        ],
        "endpoints": {
            "/health": "Server health check",
            "/enhance_wedding_ring_advanced": "🔥 Main endpoint - Lightroom style system",
            "/enhance_wedding_ring_natural": "Natural enhancement",
            "/enhance_wedding_ring_segmented": "Legacy segmented enhancement",
            "/enhance_wedding_ring_binary": "Basic binary enhancement"
        }
    })

@app.route('/health')
def health():
    """서버 상태 확인"""
    return jsonify({"status": "healthy", "version": "5.2"})

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """🔥 메인 엔드포인트 - v5.2 라이트룸 스타일 시스템"""
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # v5.2 라이트룸 스타일 보정 적용
        enhanced_image, metal_type, lighting_type, params = enhancer.enhance(image)
        
        # JPEG 인코딩 (고품질)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
        success, encoded_image = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        if not success:
            return jsonify({"error": "Failed to encode enhanced image"}), 500
        
        # 바이너리 데이터로 직접 반환
        enhanced_bytes = io.BytesIO(encoded_image.tobytes())
        enhanced_bytes.seek(0)
        
        logger.info(f"✅ Enhanced (Lightroom Style): {metal_type} ring, {lighting_type} lighting")
        
        return send_file(
            enhanced_bytes,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='enhanced_wedding_ring_v52.jpg'
        )
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    """자연스러운 보정 (v5.2와 동일)"""
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
