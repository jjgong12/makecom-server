from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import os

app = Flask(__name__)

class WeddingRingEnhancer:
    def __init__(self):
        # v10 → v13 파라미터 (아주 조금만 수정)
        self.params = {
            'brightness': 1.18,    # v10: 1.15 → v13: 1.18 (아주 조금 증가)
            'contrast': 1.12,     # v10: 1.10 → v13: 1.12 (아주 조금 증가)
            'sharpness': 1.15,    # v10: 1.12 → v13: 1.15 (아주 조금 증가)
            'white_overlay': 0.08  # v10: 없음 → v13: 8% (아주 미묘하게 추가)
        }

    def enhance_image(self, image_data):
        try:
            # 1. 이미지 읽기
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            # 2. 크기 조정 (메모리 최적화)
            height, width = original.shape[:2]
            if width > 2048:
                ratio = 2048 / width
                new_width = 2048
                new_height = int(height * ratio)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # 3. 노이즈 제거
            denoised = cv2.bilateralFilter(original, 9, 75, 75)

            # 4. PIL 보정 (v10 기본 구조 유지)
            pil_image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            # Brightness (v10: 1.15 → v13: 1.18)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(self.params['brightness'])
            
            # Contrast (v10: 1.10 → v13: 1.12)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(self.params['contrast'])
            
            # Sharpness (v10: 1.12 → v13: 1.15)
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(self.params['sharpness'])

            # 5. OpenCV로 다시 변환
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 6. v13 추가: 아주 미묘한 하얀색 오버레이 (8%)
            if self.params['white_overlay'] > 0:
                height, width = enhanced.shape[:2]
                white_overlay = np.full((height, width, 3), 255, dtype=np.uint8)
                enhanced = cv2.addWeighted(enhanced, 1 - self.params['white_overlay'], 
                                         white_overlay, self.params['white_overlay'], 0)

            # 7. 원본과 블렌딩 (자연스러움 유지)
            enhanced = cv2.addWeighted(enhanced, 0.85, denoised, 0.15, 0)

            # 8. JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded_img = cv2.imencode('.jpg', enhanced, encode_param)
            
            return {
                'success': True,
                'image_data': encoded_img.tobytes()
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 인스턴스 생성
enhancer = WeddingRingEnhancer()

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'version': 'v13-minimal-update',
        'description': 'v10에서 파라미터만 살짝 조정',
        'changes': {
            'brightness': '1.15 → 1.18',
            'contrast': '1.10 → 1.12', 
            'sharpness': '1.12 → 1.15',
            'white_overlay': '0% → 8%'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': 'v13-minimal-update'
    })

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_wedding_ring_advanced():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'image_base64 필요'}), 400

        image_data = base64.b64decode(data['image_base64'])
        result = enhancer.enhance_image(image_data)
        
        if result['success']:
            return result['image_data'], 200, {
                'Content-Type': 'image/jpeg',
                'X-Version': 'v13-minimal'
            }
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 모든 백업 엔드포인트
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_wedding_ring_v6():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_wedding_ring_binary():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_wedding_ring_segmented():
    return enhance_wedding_ring_advanced()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_wedding_ring_natural():
    return enhance_wedding_ring_advanced()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
