from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class SimpleColorEnhancer:
    def __init__(self):
        # 28쌍 After 기반 색감 조정 파라미터
        self.color_params = {
            'natural': {
                'brightness': 1.22,        # 22% 밝기 향상
                'contrast': 1.15,          # 15% 대비 향상  
                'warmth_shift': 5,         # 약간 따뜻하게
                'saturation': 1.08,        # 8% 채도 향상
                'background_overlay': [245, 240, 235],  # After 수준 배경색
                'overlay_strength': 0.4    # 40% 배경 오버레이
            },
            'warm': {
                'brightness': 1.18,
                'contrast': 1.12,
                'warmth_shift': -3,        # 따뜻함 조금 줄이기
                'saturation': 1.05,
                'background_overlay': [250, 245, 235],
                'overlay_strength': 0.5    # 더 강한 오버레이
            },
            'cool': {
                'brightness': 1.25,
                'contrast': 1.18,
                'warmth_shift': 8,         # 따뜻함 더하기
                'saturation': 1.12,
                'background_overlay': [240, 242, 250],
                'overlay_strength': 0.3    # 약한 오버레이
            }
        }
    
    def detect_lighting(self, image):
        """간단한 조명 감지"""
        try:
            # BGR을 LAB로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # B 채널 평균으로 조명 판단
            avg_b = np.mean(b)
            
            if avg_b < 120:
                return 'cool'      # 차가운 조명
            elif avg_b > 140:
                return 'warm'      # 따뜻한 조명
            else:
                return 'natural'   # 자연광
        except:
            return 'natural'
    
    def apply_warmth_shift(self, image, shift_value):
        """색온도 조정"""
        if shift_value == 0:
            return image
        
        # LAB 색공간에서 B 채널 조정
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # B 채널 조정 (양수: 따뜻하게, 음수: 차갑게)
        b = cv2.add(b, shift_value)
        b = np.clip(b, 0, 255)
        
        # 다시 합치기
        lab_adjusted = cv2.merge([l, a, b])
        return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    def create_background_overlay(self, image, overlay_color, strength):
        """배경 오버레이로 "딸깍" 단순화"""
        height, width = image.shape[:2]
        
        # 단색 배경 생성
        overlay = np.full((height, width, 3), overlay_color, dtype=np.uint8)
        
        # 중앙에서 바깥쪽으로 그라데이션 마스크
        center_y, center_x = height // 2, width // 2
        
        # 거리 기반 마스크 (중앙은 약하게, 가장자리는 강하게)
        Y, X = np.ogrid[:height, :width]
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 정규화된 거리 (0: 중앙, 1: 가장자리)
        normalized_distances = distances / max_distance
        
        # 가장자리로 갈수록 더 강한 오버레이
        distance_mask = np.clip(normalized_distances * strength + 0.1, 0, strength)
        distance_mask = np.stack([distance_mask, distance_mask, distance_mask], axis=-1)
        
        # 오버레이 적용
        result = image * (1 - distance_mask) + overlay * distance_mask
        return result.astype(np.uint8)
    
    def simple_enhance(self, image_data):
        """단순 전체 색감 조정"""
        try:
            # 이미지 디코딩 및 리샘플링
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            height, width = original.shape[:2]
            if height > 2048 or width > 2048:
                scale = min(2048/height, 2048/width)
                new_height, new_width = int(height * scale), int(width * scale)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 기본 노이즈 제거
            denoised = cv2.bilateralFilter(original, 9, 75, 75)
            
            # 1단계: 조명 감지
            lighting = self.detect_lighting(denoised)
            params = self.color_params[lighting]
            
            # 2단계: PIL 기반 기본 보정
            img_pil = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            # 밝기 조정
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(params['brightness'])
            
            # 대비 조정
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(params['contrast'])
            
            # 채도 조정
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(params['saturation'])
            
            # PIL → OpenCV
            enhanced = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 3단계: 색온도 조정
            enhanced = self.apply_warmth_shift(enhanced, params['warmth_shift'])
            
            # 4단계: 배경 "딸깍" 오버레이
            enhanced = self.create_background_overlay(
                enhanced, 
                params['background_overlay'], 
                params['overlay_strength']
            )
            
            # 5단계: 제한적 CLAHE (디테일 살리기)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 6단계: 최종 블렌딩 (90% 보정 + 10% 원본)
            final_result = cv2.addWeighted(enhanced, 0.9, denoised, 0.1, 0)
            
            return final_result, lighting
            
        except Exception as e:
            logging.error(f"Enhancement error: {str(e)}")
            return None, "error"

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "Simple Color v1.0",
        "message": "단순 전체 색감 조정 - 마스킹 없는 깔끔한 처리",
        "endpoints": [
            "/health",
            "/enhance_wedding_ring_v6",
            "/enhance_wedding_ring_simple"
        ]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "Simple Color v1.0"})

@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 단순 색감 조정
        enhancer = SimpleColorEnhancer()
        enhanced_image, lighting = enhancer.simple_enhance(image_data)
        
        if enhanced_image is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_wedding_ring_simple', methods=['POST'])
def enhance_wedding_ring_simple():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(data['image_base64'])
        
        # 단순 색감 조정
        enhancer = SimpleColorEnhancer()
        enhanced_image, lighting = enhancer.simple_enhance(image_data)
        
        if enhanced_image is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', enhanced_image, encode_param)
        
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
