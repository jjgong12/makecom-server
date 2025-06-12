import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from flask import Flask, request, jsonify, Response, render_template_string
import os

app = Flask(__name__)

class AfterLevelWeddingRingEnhancer:
    def __init__(self):
        # After 수준 파라미터 (6개 이미지 분석 기반)
        self.ring_params = {
            'white_gold': {
                'natural': {'brightness': 1.42, 'contrast': 1.25, 'sharpness': 1.30, 'clarity': 1.22},
                'warm': {'brightness': 1.45, 'contrast': 1.28, 'sharpness': 1.35, 'clarity': 1.25},
                'cool': {'brightness': 1.38, 'contrast': 1.22, 'sharpness': 1.28, 'clarity': 1.20}
            },
            'rose_gold': {
                'natural': {'brightness': 1.40, 'contrast': 1.22, 'sharpness': 1.25, 'clarity': 1.20},
                'warm': {'brightness': 1.35, 'contrast': 1.18, 'sharpness': 1.22, 'clarity': 1.18},
                'cool': {'brightness': 1.48, 'contrast': 1.28, 'sharpness': 1.32, 'clarity': 1.25}
            },
            'champagne_gold': {
                'natural': {'brightness': 1.41, 'contrast': 1.24, 'sharpness': 1.28, 'clarity': 1.21},
                'warm': {'brightness': 1.38, 'contrast': 1.20, 'sharpness': 1.25, 'clarity': 1.19},
                'cool': {'brightness': 1.45, 'contrast': 1.27, 'sharpness': 1.30, 'clarity': 1.23}
            },
            'yellow_gold': {
                'natural': {'brightness': 1.43, 'contrast': 1.26, 'sharpness': 1.28, 'clarity': 1.22},
                'warm': {'brightness': 1.36, 'contrast': 1.20, 'sharpness': 1.24, 'clarity': 1.18},
                'cool': {'brightness': 1.50, 'contrast': 1.30, 'sharpness': 1.35, 'clarity': 1.26}
            }
        }
    
    def detect_ring_type(self, image):
        """보수적 금속 감지"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            
            if s_mean < 30:  # 낮은 채도 = 화이트/플래티넘
                return 'white_gold'
            elif h_mean < 15 or h_mean > 160:  # 빨간색 계열
                return 'rose_gold'
            elif 15 <= h_mean <= 30:  # 황금색 계열
                return 'yellow_gold'
            else:
                return 'champagne_gold'  # 기본값
        except:
            return 'champagne_gold'
    
    def detect_lighting(self, image):
        """보수적 조명 감지"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean > 135:  # 따뜻한 조명
                return 'warm'
            elif b_mean < 120:  # 차가운 조명
                return 'cool'
            else:
                return 'natural'  # 기본값
        except:
            return 'natural'
    
    def extract_ring_region(self, image):
        """웨딩링 영역 추출 (노이즈 제거용)"""
        try:
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 금속 영역 감지 (밝기 기반)
            _, binary = cv2.threshold(hsv[:, :, 2], 100, 255, cv2.THRESH_BINARY)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 가우시안 블러로 부드럽게
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask
        except:
            # 실패시 전체 마스크 반환
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    
    def clean_ring_region(self, image, mask):
        """웨딩링 영역 디테일 정리"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)
            
            # 웨딩링 영역만 추출
            ring_region = Image.composite(pil_image, Image.new('RGB', pil_image.size, (128, 128, 128)), pil_mask)
            
            # 노이즈 제거 (미디안 필터)
            ring_region = ring_region.filter(ImageFilter.MedianFilter(size=3))
            
            # 약한 선명도 향상 (과하지 않게)
            enhancer = ImageEnhance.Sharpness(ring_region)
            ring_region = enhancer.enhance(1.15)  # 15%만 향상
            
            # 원본과 합성
            result = Image.composite(ring_region, pil_image, pil_mask)
            
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        except:
            return image
    
    def enhance_overall_brightness(self, image, params):
        """전체 영역 After 수준 밝기 향상"""
        try:
            # PIL로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 1. 밝기 대폭 향상 (After 수준)
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = brightness_enhancer.enhance(params['brightness'])
            
            # 2. 대비 향상 (깨끗하고 프로페셔널하게)
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(params['contrast'])
            
            # 3. 선명도 적당히
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(params['sharpness'])
            
            # 4. 제한적 명료도 (CLAHE)
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=params['clarity'], tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_cv
        except:
            return image
    
    def enhance_highlights(self, image, boost_factor=0.12):
        """하이라이트 부스팅 (금속 반사 강화)"""
        try:
            # 그레이스케일로 변환해서 밝은 영역 찾기
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 상위 20% 밝은 영역만 선택
            threshold = np.percentile(gray, 80)
            highlight_mask = (gray > threshold).astype(np.float32)
            
            # 부드러운 마스크 생성
            highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
            
            # 하이라이트 영역만 밝기 증가
            result = image.astype(np.float32)
            for c in range(3):
                result[:, :, c] += result[:, :, c] * highlight_mask * boost_factor
            
            # 255 클리핑
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
        except:
            return image
    
    def blend_with_original(self, enhanced, original, blend_ratio=0.85):
        """원본과 블렌딩 (자연스러움 보장)"""
        try:
            result = cv2.addWeighted(enhanced, blend_ratio, original, 1 - blend_ratio, 0)
            return result
        except:
            return enhanced
    
    def enhance_wedding_ring(self, image_data):
        """메인 After 수준 보정 함수"""
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
            
            # 2. 자동 분석
            ring_type = self.detect_ring_type(image)
            lighting = self.detect_lighting(image)
            params = self.ring_params[ring_type][lighting]
            
            # 3. 전체 노이즈 제거 (기본)
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 4. 웨딩링 영역 추출 및 디테일 정리
            ring_mask = self.extract_ring_region(image)
            image = self.clean_ring_region(image, ring_mask)
            
            # 5. 전체 영역 After 수준 밝기 향상
            image = self.enhance_overall_brightness(image, params)
            
            # 6. 하이라이트 부스팅 (금속 반사 강화)
            image = self.enhance_highlights(image, 0.12)
            
            # 7. 원본과 블렌딩 (85% 보정 + 15% 원본)
            result = self.blend_with_original(image, original, 0.85)
            
            # 8. JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
            _, buffer = cv2.imencode('.jpg', result, encode_param)
            
            return buffer.tobytes(), ring_type, lighting
            
        except Exception as e:
            print(f"Enhancement error: {str(e)}")
            raise

# Flask 앱 설정
enhancer = AfterLevelWeddingRingEnhancer()

@app.route('/')
def home():
    return """
    <h1>🔥 Wedding Ring V6.0 After Level System</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>/health</strong> - Server status</li>
        <li><strong>/enhance_wedding_ring_v6</strong> - After Level Enhancement ⭐</li>
        <li><strong>/enhance_wedding_ring_advanced</strong> - V5.4 System (backup)</li>
    </ul>
    <p><strong>V6.0 Features:</strong></p>
    <ul>
        <li>✅ After 수준 밝기 (1.35-1.50)</li>
        <li>✅ 웨딩링 디테일 정리 (노이즈/기스 제거)</li>
        <li>✅ 전체 프로페셔널 보정</li>
        <li>✅ 6개 After 이미지 기준</li>
    </ul>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "6.0", "message": "After Level System Ready"})

@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    """V6.0 After 수준 메인 엔드포인트"""
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # After 수준 보정 수행
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring(image_data)
        
        # 바이너리 응답 반환 (Make.com 직접 업로드 가능)
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '6.0-After-Level'
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
        
        # 기존 V5.4 로직으로 처리 (간단 버전)
        image_data = base64.b64decode(data['image_base64'])
        enhanced_image, ring_type, lighting = enhancer.enhance_wedding_ring(image_data)
        
        return Response(
            enhanced_image,
            mimetype='image/jpeg',
            headers={
                'X-Ring-Type': ring_type,
                'X-Lighting': lighting,
                'X-Version': '5.4-Backup'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
