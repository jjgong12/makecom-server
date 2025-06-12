import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from flask import Flask, request, jsonify, Response, render_template_string
import os

app = Flask(__name__)

class DetailPreservingEnhancer:
    def __init__(self):
        # 배경 영역 파라미터 (더 보수적, 원본에 가깝게)
        self.background_params = {
            'white_gold': {
                'natural': {'brightness': 1.15, 'contrast': 1.08, 'sharpness': 1.10, 'clarity': 1.05},
                'warm': {'brightness': 1.18, 'contrast': 1.10, 'sharpness': 1.12, 'clarity': 1.07},
                'cool': {'brightness': 1.12, 'contrast': 1.06, 'sharpness': 1.08, 'clarity': 1.04}
            },
            'rose_gold': {
                'natural': {'brightness': 1.14, 'contrast': 1.07, 'sharpness': 1.09, 'clarity': 1.05},
                'warm': {'brightness': 1.10, 'contrast': 1.05, 'sharpness': 1.07, 'clarity': 1.03},
                'cool': {'brightness': 1.20, 'contrast': 1.12, 'sharpness': 1.15, 'clarity': 1.08}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.16, 'contrast': 1.09, 'sharpness': 1.10, 'clarity': 1.06},
                'warm': {'brightness': 1.13, 'contrast': 1.07, 'sharpness': 1.08, 'clarity': 1.04},
                'cool': {'brightness': 1.19, 'contrast': 1.11, 'sharpness': 1.13, 'clarity': 1.07}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.17, 'contrast': 1.10, 'sharpness': 1.11, 'clarity': 1.06},
                'warm': {'brightness': 1.11, 'contrast': 1.06, 'sharpness': 1.07, 'clarity': 1.03},
                'cool': {'brightness': 1.22, 'contrast': 1.14, 'sharpness': 1.16, 'clarity': 1.09}
            }
        }
        
        # 웨딩링 영역 파라미터 (확대샷 수준, 디테일 보존)
        self.ring_params = {
            'white_gold': {
                'natural': {'brightness': 1.50, 'contrast': 1.30, 'sharpness': 1.45, 'clarity': 1.35},
                'warm': {'brightness': 1.55, 'contrast': 1.35, 'sharpness': 1.50, 'clarity': 1.40},
                'cool': {'brightness': 1.45, 'contrast': 1.25, 'sharpness': 1.40, 'clarity': 1.30}
            },
            'rose_gold': {
                'natural': {'brightness': 1.48, 'contrast': 1.28, 'sharpness': 1.42, 'clarity': 1.32},
                'warm': {'brightness': 1.43, 'contrast': 1.23, 'sharpness': 1.38, 'clarity': 1.28},
                'cool': {'brightness': 1.58, 'contrast': 1.38, 'sharpness': 1.55, 'clarity': 1.45}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.52, 'contrast': 1.32, 'sharpness': 1.46, 'clarity': 1.36},
                'warm': {'brightness': 1.46, 'contrast': 1.26, 'sharpness': 1.40, 'clarity': 1.30},
                'cool': {'brightness': 1.60, 'contrast': 1.40, 'sharpness': 1.58, 'clarity': 1.48}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.54, 'contrast': 1.34, 'sharpness': 1.48, 'clarity': 1.38},
                'warm': {'brightness': 1.44, 'contrast': 1.24, 'sharpness': 1.36, 'clarity': 1.26},
                'cool': {'brightness': 1.65, 'contrast': 1.45, 'sharpness': 1.65, 'clarity': 1.55}
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
    
    def create_precision_ring_mask(self, image):
        """정밀한 웨딩링 마스크 생성 (디테일 보존용)"""
        try:
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 더 정교한 금속 영역 감지
            # 조건 1: 밝기 기반 (더 넓은 범위)
            _, bright_mask1 = cv2.threshold(hsv[:, :, 2], 70, 255, cv2.THRESH_BINARY)
            _, bright_mask2 = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY_INV)
            bright_combined = cv2.bitwise_and(bright_mask1, bright_mask2)
            
            # 조건 2: 채도 기반 (금속의 특성)
            _, sat_mask1 = cv2.threshold(hsv[:, :, 1], 15, 255, cv2.THRESH_BINARY)
            _, sat_mask2 = cv2.threshold(hsv[:, :, 1], 180, 255, cv2.THRESH_BINARY_INV)
            sat_combined = cv2.bitwise_and(sat_mask1, sat_mask2)
            
            # 두 조건 결합
            combined_mask = cv2.bitwise_and(bright_combined, sat_combined)
            
            # 정교한 모폴로지 연산
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            
            # 작은 노이즈 제거
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
            # 연결 강화
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
            
            # 디테일 보존을 위한 적당한 블러 (너무 강하지 않게)
            combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)
            
            # 0-1 범위로 정규화
            mask_normalized = combined_mask.astype(np.float32) / 255.0
            
            return mask_normalized
        except:
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def enhance_with_detail_preservation(self, image, params):
        """디테일 보존하며 파라미터 기반 보정"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 1. 밝기 향상 (확대샷 수준)
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 2. 대비 강화 (디테일 살리기)
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 3. 선명도 강화 (디테일 보존)
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # OpenCV로 변환
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            # 4. 고강도 CLAHE (디테일 극대화)
            lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=params['clarity'], tileGridSize=(8, 8))  # 더 작은 타일로 디테일 강화
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_cv
        except:
            return image
    
    def selective_detail_enhancement(self, image, ring_mask, background_params, ring_params):
        """영역별 선택적 디테일 보정"""
        try:
            # 배경 영역 보정 (보수적)
            background_enhanced = self.enhance_with_detail_preservation(image, background_params)
            
            # 웨딩링 영역 보정 (확대샷 수준 + 디테일 강화)
            ring_enhanced = self.enhance_with_detail_preservation(image, ring_params)
            
            # 정밀한 블렌딩
            result = background_enhanced.astype(np.float32)
            ring_enhanced_f = ring_enhanced.astype(np.float32)
            
            # 3채널 마스크 확장
            mask_3d = np.stack([ring_mask, ring_mask, ring_mask], axis=2)
            
            # 웨딩링 영역 선택적 적용
            result = result * (1 - mask_3d) + ring_enhanced_f * mask_3d
            
            # 정수형 변환
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
        except:
            return image
    
    def advanced_noise_reduction(self, image):
        """고급 노이즈 제거 (디테일 보존)"""
        try:
            # 디테일 보존형 bilateral filter
            result = cv2.bilateralFilter(image, 7, 80, 80)
            return result
        except:
            return image
    
    def detail_aware_highlight_boost(self, image, boost_factor=0.08):
        """디테일 인식 하이라이트 부스팅"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 상위 20% 밝은 영역 (더 선택적)
            threshold = np.percentile(gray, 80)
            highlight_mask = (gray > threshold).astype(np.float32)
            
            # 적당한 블러 (디테일 보존)
            highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
            
            # 8% 하이라이트 증가 (확대샷 수준)
            result = image.astype(np.float32)
            for c in range(3):
                result[:, :, c] += result[:, :, c] * highlight_mask * boost_factor
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        except:
            return image
    
    def conservative_blend_with_original(self, enhanced, original, blend_ratio=0.80):
        """더 보수적인 원본 블렌딩"""
        try:
            # 80% 보정 + 20% 원본 (더 자연스럽게)
            result = cv2.addWeighted(enhanced, blend_ratio, original, 1 - blend_ratio, 0)
            return result
        except:
            return enhanced
    
    def enhance_wedding_ring_with_details(self, image_data):
        """디테일 보존 + 확대샷 수준 메인 함수"""
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
            
            # 3. 고급 노이즈 제거 (디테일 보존)
            image = self.advanced_noise_reduction(image)
            
            # 4. 정밀한 웨딩링 마스크 생성
            ring_mask = self.create_precision_ring_mask(image)
            
            # 5. 영역별 선택적 디테일 보정 (핵심)
            image = self.selective_detail_enhancement(image, ring_mask, background_params, ring_params)
            
            # 6. 디테일 인식 하이라이트 부스팅
            image = self.detail_aware_highlight_boost(image, 0.08)
            
            # 7. 보수적 원본 블렌딩
            result = self.conservative_blend_with_original(image, original, 0.80)
            
            # 8. JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            _, buffer = cv2.imencode('.jpg', result, encode_param)
            
            return buffer.tobytes(), ring_type, lighting
            
        except Exception as e:
            print(f"Enhancement error: {str(e)}")
            raise

# Flask 앱 설정
enhancer = DetailPreservingEnhancer()

@app.route('/')
def home():
    return """
    <h1>🔥 Wedding Ring V6.3 Detail Preserving System</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>/health</strong> - Server status</li>
        <li><strong>/enhance_wedding_ring_v6</strong> - Detail Preserving Enhancement ⭐</li>
        <li><strong>/enhance_wedding_ring_advanced</strong> - V6.2 System (backup)</li>
    </ul>
    <p><strong>V6.3 Features:</strong></p>
    <ul>
        <li>✅ 배경: 원본에 더 가까움 (1.10-1.22)</li>
        <li>✅ 웨딩링: 확대샷 수준 밝기 (1.45-1.65)</li>
        <li>✅ 디테일 극대화 (밀그레인, 큐빅, 텍스처)</li>
        <li>✅ 정밀한 마스크 + 보수적 블렌딩</li>
    </ul>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "6.3", "message": "Detail Preserving System Ready"})

@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """V6.3 디테일 보존 메인 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 디테일 보존 보정 수행
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_with_details(image_data)
        
        # 바이너리 응답 반환
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.3-Detail-Preserving'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    """V6.2 백업 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # V6.3 로직으로 처리 (동일)
        image_data = base64.b64decode(data['image_base64'])
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring_with_details(image_data)
        
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.3-Backup'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
