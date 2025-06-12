import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from flask import Flask, request, jsonify, Response, render_template_string
import os

app = Flask(__name__)

class SelectiveBrightnessEnhancer:
    def __init__(self):
        # 배경 영역 파라미터 (v6.1 수준 유지)
        self.background_params = {
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
        
        # 웨딩링 영역 파라미터 (더 밝게)
        self.ring_params = {
            'white_gold': {
                'natural': {'brightness': 1.38, 'contrast': 1.22, 'sharpness': 1.25, 'clarity': 1.18},
                'warm': {'brightness': 1.42, 'contrast': 1.25, 'sharpness': 1.28, 'clarity': 1.20},
                'cool': {'brightness': 1.35, 'contrast': 1.20, 'sharpness': 1.22, 'clarity': 1.15}
            },
            'rose_gold': {
                'natural': {'brightness': 1.37, 'contrast': 1.21, 'sharpness': 1.23, 'clarity': 1.16},
                'warm': {'brightness': 1.33, 'contrast': 1.17, 'sharpness': 1.20, 'clarity': 1.14},
                'cool': {'brightness': 1.43, 'contrast': 1.27, 'sharpness': 1.28, 'clarity': 1.21}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.39, 'contrast': 1.23, 'sharpness': 1.24, 'clarity': 1.17},
                'warm': {'brightness': 1.36, 'contrast': 1.20, 'sharpness': 1.22, 'clarity': 1.15},
                'cool': {'brightness': 1.42, 'contrast': 1.26, 'sharpness': 1.27, 'clarity': 1.20}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.40, 'contrast': 1.24, 'sharpness': 1.25, 'clarity': 1.18},
                'warm': {'brightness': 1.34, 'contrast': 1.18, 'sharpness': 1.21, 'clarity': 1.14},
                'cool': {'brightness': 1.45, 'contrast': 1.29, 'sharpness': 1.31, 'clarity': 1.22}
            }
        }
    
    def detect_ring_type(self, image):
        """보수적 금속 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            if s_mean < 25:
                return 'white_gold'
            elif h_mean < 15 or h_mean > 160:
                return 'rose_gold'
            elif 15 <= h_mean <= 30:
                return 'yellow_gold'
            else:
                return 'champagne_gold'
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean > 135:
                return 'warm'
            elif b_mean < 115:
                return 'cool'
            else:
                return 'natural'
        except:
            return 'natural'
    
    def create_smooth_ring_mask(self, image):
        """부드러운 웨딩링 마스크 생성 (경계선 문제 해결)"""
        try:
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 금속 영역 감지 (다중 조건)
            # 조건 1: 밝기 기반
            _, bright_mask = cv2.threshold(hsv[:, :, 2], 80, 255, cv2.THRESH_BINARY)
            
            # 조건 2: 채도 기반 (금속은 보통 중간 채도)
            _, sat_mask = cv2.threshold(hsv[:, :, 1], 30, 255, cv2.THRESH_BINARY)
            sat_mask2 = cv2.threshold(hsv[:, :, 1], 200, 255, cv2.THRESH_BINARY_INV)[1]
            sat_combined = cv2.bitwise_and(sat_mask, sat_mask2)
            
            # 두 조건 결합
            combined_mask = cv2.bitwise_and(bright_mask, sat_combined)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # 매우 부드러운 가우시안 블러 (경계선 완전 제거)
            combined_mask = cv2.GaussianBlur(combined_mask, (31, 31), 0)
            
            # 0-1 범위로 정규화
            mask_normalized = combined_mask.astype(np.float32) / 255.0
            
            return mask_normalized
        except:
            # 실패시 전체 마스크 반환
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def enhance_with_params(self, image, params):
        """파라미터 기반 보정"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 밝기 향상
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 대비 향상
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 선명도 향상
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # OpenCV로 변환
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            # 제한적 CLAHE
            lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=params['clarity'], tileGridSize=(16, 16))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_cv
        except:
            return image
    
    def selective_enhancement(self, image, ring_mask, background_params, ring_params):
        """영역별 선택적 보정"""
        try:
            # 배경 영역 보정
            background_enhanced = self.enhance_with_params(image, background_params)
            
            # 웨딩링 영역 보정 (더 밝게)
            ring_enhanced = self.enhance_with_params(image, ring_params)
            
            # 부드러운 블렌딩 (마스크 기반)
            result = background_enhanced.astype(np.float32)
            ring_enhanced_f = ring_enhanced.astype(np.float32)
            
            # 3채널 마스크 확장
            mask_3d = np.stack([ring_mask, ring_mask, ring_mask], axis=2)
            
            # 웨딩링 영역만 선택적으로 밝게
            result = result * (1 - mask_3d) + ring_enhanced_f * mask_3d
            
            # 정수형으로 변환
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
        except:
            return image
    
    def gentle_noise_reduction(self, image):
        """부드러운 노이즈 제거"""
        try:
            result = cv2.bilateralFilter(image, 5, 50, 50)
            return result
        except:
            return image
    
    def subtle_highlight_boost(self, image, boost_factor=0.05):
        """미묘한 하이라이트 부스팅"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = np.percentile(gray, 75)
            highlight_mask = (gray > threshold).astype(np.float32)
            highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
            
            result = image.astype(np.float32)
            for c in range(3):
                result[:, :, c] += result[:, :, c] * highlight_mask * boost_factor
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        except:
            return image
    
    def safe_blend_with_original(self, enhanced, original, blend_ratio=0.75):
        """안전한 원본 블렌딩"""
        try:
            result = cv2.addWeighted(enhanced, blend_ratio, original, 1 - blend_ratio, 0)
            return result
        except:
            return enhanced
    
    def enhance_wedding_ring_selective(self, image_data):
        """선택적 밝기 강화 메인 함수"""
        try:
            # 1. 이미지 디코딩 및 리사이징
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # 메모리 최적화
            height, width = image.shape[:2]
            if width > 2048:
                scale = 2048 / width
                new_width = 2048
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            original = image.copy()
            
            # 2. 자동 분석
            ring_type = self.detect_ring_type(image)
            lighting = self.detect_lighting(image)
            background_params = self.background_params[ring_type][lighting]
            ring_params = self.ring_params[ring_type][lighting]
            
            # 3. 부드러운 노이즈 제거
            image = self.gentle_noise_reduction(image)
            
            # 4. 부드러운 웨딩링 마스크 생성
            ring_mask = self.create_smooth_ring_mask(image)
            
            # 5. 영역별 선택적 보정 (핵심)
            image = self.selective_enhancement(image, ring_mask, background_params, ring_params)
            
            # 6. 미묘한 하이라이트 부스팅
            image = self.subtle_highlight_boost(image, 0.05)
            
            # 7. 원본과 블렌딩
            result = self.safe_blend_with_original(image, original, 0.75)
            
            # 8. JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            _, buffer = cv2.imencode('.jpg', result, encode_param)
            
            return buffer.tobytes(), ring_type, lighting
            
        except Exception as e:
            print(f"Enhancement error: {str(e)}")
            raise

# Flask 앱 설정
enhancer = SelectiveBrightnessEnhancer()

@app.route('/')
def home():
    return """
    <h1>🔥 Wedding Ring V6.2 Selective Brightness System</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>/health</strong> - Server status</li>
        <li><strong>/enhance_wedding_ring_v6</strong> - Selective Brightness Enhancement ⭐</li>
        <li><strong>/enhance_wedding_ring_advanced</strong> - V6.1 System (backup)</li>
    </ul>
    <p><strong>V6.2 Features:</strong></p>
    <ul>
        <li>✅ 배경: v6.1 수준 유지 (자연스럽고 깨끗함)</li>
        <li>✅ 웨딩링: 15% 더 밝게 (확대샷 수준)</li>
        <li>✅ 부드러운 경계 블렌딩 (그라데이션 없음)</li>
        <li>✅ 영역별 차별 보정</li>
    </ul>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "6.2", "message": "Selective Brightness System Ready"})

@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """V6.2 선택적 밝기 강화 메인 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 선택적 밝기 강화 수행
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_selective(image_data)
        
        # 바이너리 응답 반환
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.2-Selective-Brightness'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """V6.1 백업 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # V6.2 로직으로 처리 (동일)
        image_data = base64.b64decode(data['image_base64'])
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_selective(image_data)
        
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.2-Backup'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
