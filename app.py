from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import os
import numpy as np

app = Flask(__name__)

class SimpleWeddingRingEnhancer:
    def __init__(self):
        # 단순화된 파라미터 (v13.3 유지)
        self.params = {
            'brightness': 1.18,
            'contrast': 1.12,
            'sharpness': 1.15
        }

    def enhance_image(self, image_data):
        try:
            # Base64 → PIL
            image = Image.open(io.BytesIO(image_data))
            
            # RGB 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 크기 조정 (메모리 절약)
            width, height = image.size
            if width > 2048:
                ratio = 2048 / width
                new_width = 2048
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # PIL 보정만 사용
            # 1. Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.params['brightness'])
            
            # 2. Contrast  
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.params['contrast'])
            
            # 3. Sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(self.params['sharpness'])
            
            # 4. 노이즈 제거 (PIL 필터)
            image = image.filter(ImageFilter.SMOOTH_MORE)
            
            # JPEG 저장
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95, optimize=True)
            
            return {
                'success': True,
                'image_data': output.getvalue()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 인스턴스 생성
enhancer = SimpleWeddingRingEnhancer()

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'version': 'ultra-minimal',
        'message': 'PIL만 사용하는 안전한 버전'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': 'ultra-minimal'
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
                'X-Version': 'ultra-minimal'
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
