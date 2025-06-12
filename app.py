import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from flask import Flask, request, jsonify, Response, render_template_string
import os

app = Flask(__name__)

class NaturalAfterLevelEnhancer:
    def __init__(self):
        # 자연스러운 After 수준 파라미터 (마지막 이미지 기준)
        self.ring_params = {
            'white_gold': {
                'natural': {'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.18, 'clarity': 1.10},
                'warm': {'brightness': 1.28, 'contrast': 1.18, 'sharpness': 1.20, 'clarity': 1.12},
                'cool': {'brightness': 1.22, 'contrast': 1.12, 'sharpness': 1.15, 'clarity': 1.08}
            },
            'rose_gold': {
                'natural': {'brightness': 1.24, 'contrast': 1.14, 'sharpness': 1.16, 'clarity': 1.09},
                'warm': {'brightness': 1.20, 'contrast': 1.10, 'sharpness': 1.14, 'clarity': 1.07},
                'cool': {'brightness': 1.30, 'contrast': 1.20, 'sharpness': 1.22, 'clarity': 1.14}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.26, 'contrast': 1.16, 'sharpness': 1.17, 'clarity': 1.10},
                'warm': {'brightness': 1.23, 'contrast': 1.13, 'sharpness': 1.15, 'clarity': 1.08},
                'cool': {'brightness': 1.29, 'contrast': 1.19, 'sharpness': 1.20, 'clarity': 1.13}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.27, 'contrast': 1.17, 'sharpness': 1.18, 'clarity': 1.11},
                'warm': {'brightness': 1.21, 'contrast': 1.11, 'sharpness': 1.14, 'clarity': 1.07},
                'cool': {'brightness': 1.32, 'contrast': 1.22, 'sharpness': 1.24, 'clarity': 1.15}
            }
        }
    
    def detect_ring_type(self, image):
        """보수적 금속 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            if s_mean < 25:  # 낮은 채도 = 화이트골드
                return 'white_gold'
            elif h_mean < 15 or h_mean > 160:  # 빨간색 계열
                return 'rose_gold'
            elif 15 <= h_mean <= 30:  # 황금색 계열
                return 'yellow_gold'
            else:
                return 'champagne_gold'  # 기본값 (가장 안전)
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean > 135:  # 따뜻한 조명
                return 'warm'
            elif b_mean < 115:  # 차가운 조명
                return 'cool'
            else:
                return 'natural'  # 기본값 (가장 안전)
        except:
            return 'natural'
    
    def gentle_noise_reduction(self, image):
        """부드러운 노이즈 제거"""
        try:
            # 매우 부드러운 bilateral filter
            result = cv2.bilateralFilter(image, 5, 50, 50)
            return result
        except:
            return image
    
    def enhance_ring_details(self, image):
        """웨딩링 디테일 향상 (최소한만)"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 매우 약한 선명도 향상 (10%만)
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced = enhancer.enhance(1.10)
            
            # 다시 OpenCV로 변환
            result = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            return result
        except:
            return image
    
    def natural_brightness_enhancement(self, image, params):
        """자연스러운 전체 밝기 향상"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 1. 적당한 밝기 향상 (과하지 않게)
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 2. 부드러운 대비 향상
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 3. 약한 선명도
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # OpenCV로 변환
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            # 4. 매우 제한적 CLAHE (자연스럽게)
            lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=params['clarity'], tileGridSize=(16, 16))  # 더 큰 타일로 부드럽게
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_cv
        except:
            return image
    
    def subtle_highlight_boost(self, image, boost_factor=0.05):
        """매우 미묘한 하이라이트 부스팅"""
        try:
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 상위 25% 밝은 영역만 선택 (더 제한적)
            threshold = np.percentile(gray, 75)
            highlight_mask = (gray > threshold).astype(np.float32)
            
            # 매우 부드러운 마스크 생성
            highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
            
            # 매우 미묘한 하이라이트 증가 (5%만)
            result = image.astype(np.float32)
            for c in range(3):
                result[:, :, c] += result[:, :, c] * highlight_mask * boost_factor
            
            # 255 클리핑
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
        except:
            return image
    
    def safe_blend_with_original(self, enhanced, original, blend_ratio=0.75):
        """안전한 원본 블렌딩 (더 보수적)"""
        try:
            # 75% 보정 + 25% 원본 (더 자연스럽게)
            result = cv2.addWeighted(enhanced, blend_ratio, original, 1 - blend_ratio, 0)
            return result
        except:
            return enhanced
    
    def enhance_wedding_ring_natural(self, image_data):
        """자연스러운 After 수준 메인 함수"""
        try:
            # 1. 이미지 디코딩 및 리사이징
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # 메모리 최적화를 위한 리사이징
            height, width = image.shape[:2]
            if width > 2048:
                scale = 2048 / width
                new_width = 2048
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            original = image.copy()
            
            # 2. 자동 분석 (보수적)
            ring_type = self.detect_ring_type(image)
            lighting = self.detect_lighting(image)
            params = self.ring_params[ring_type][lighting]
            
            # 3. 부드러운 노이즈 제거
            image = self.gentle_noise_reduction(image)
            
            # 4. 웨딩링 디테일 최소한 향상
            image = self.enhance_ring_details(image)
            
            # 5. 자연스러운 전체 밝기 향상 (핵심)
            image = self.natural_brightness_enhancement(image, params)
            
            # 6. 매우 미묘한 하이라이트 부스팅
            image = self.subtle_highlight_boost(image, 0.05)
            
            # 7. 보수적 원본 블렌딩 (75% 보정 + 25% 원본)
            result = self.safe_blend_with_original(image, original, 0.75)
            
            # 8. JPEG 인코딩 (고품질)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            _, buffer = cv2.imencode('.jpg', result, encode_param)
            
            return buffer.tobytes(), ring_type, lighting
            
        except Exception as e:
            print(f"Enhancement error: {str(e)}")
            raise

# Flask 앱 설정
enhancer = NaturalAfterLevelEnhancer()

@app.route('/')
def home():
    return """
    <h1>🔥 Wedding Ring V6.1 Natural After Level System</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>/health</strong> - Server status</li>
        <li><strong>/enhance_wedding_ring_v6</strong> - Natural After Level ⭐</li>
        <li><strong>/enhance_wedding_ring_advanced</strong> - V5.4 System (backup)</li>
    </ul>
    <p><strong>V6.1 Features:</strong></p>
    <ul>
        <li>✅ 자연스러운 After 수준 (1.22-1.32)</li>
        <li>✅ 균등하고 부드러운 전체 보정</li>
        <li>✅ 미묘한 디테일 향상</li>
        <li>✅ 과도함 방지 (75% 블렌딩)</li>
        <li>✅ 그라데이션 효과 완전 제거</li>
    </ul>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "6.1", "message": "Natural After Level System Ready"})

@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """V6.1 자연스러운 After 수준 메인 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 자연스러운 After 수준 보정 수행
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_natural(image_data)
        
        # 바이너리 응답 반환
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.1-Natural-After-Level'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """V5.4 백업 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # V6.1 로직으로 처리 (동일)
        image_data = base64.b64decode(data['image_base64'])
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_natural(image_data)
        
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.1-Backup'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
