from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import os

app = Flask(__name__)

class PILWeddingRingEnhancer:
    def __init__(self):
        # v13.3 파라미터 (버전10 쪽 조정) 유지
        self.params = {
            'brightness': 1.18,    # 버전10 쪽 조정값
            'contrast': 1.12,     # 버전10 쪽 조정값  
            'sharpness': 1.15,    # 버전10 쪽 조정값
            'white_blend': 0.09   # 9% 하얀색 블렌딩
        }

    def enhance_image(self, image_data):
        try:
            # 이미지 로드
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
            
            # 원본 보존
            original = image.copy()
            
            # 1. Brightness (v13.3 파라미터)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.params['brightness'])
            
            # 2. Contrast (v13.3 파라미터)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.params['contrast'])
            
            # 3. Sharpness (v13.3 파라미터)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(self.params['sharpness'])
            
            # 4. 하얀색 오버레이 (v13.3 9%)
            white_overlay = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.blend(image, white_overlay, self.params['white_blend'])
            
            # 5. 원본과 블렌딩 (15% 원본 보존)
            image = Image.blend(image, original, 0.15)
            
            # 6. 부드러운 노이즈 제거
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
enhancer = PILWeddingRingEnhancer()

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'version': 'v13.3-PIL-only',
        'description': 'PIL만 사용, v13.3 파라미터 유지',
        'params': {
            'brightness': 1.18,
            'contrast': 1.12, 
            'sharpness': 1.15,
            'white_blend': '9%'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': 'v13.3-PIL-only',
        'message': 'OpenCV 제거, PIL만 사용'
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
                'X-Version': 'v13.3-PIL-only',
                'X-Params': 'brightness-1.18_contrast-1.12_sharpness-1.15'
            }
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 모든 백업 엔드포인트 (호환성)
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
