from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
import base64
import traceback
from datetime import datetime

app = Flask(__name__)

class CombinedABEnhancer:
    def __init__(self):
        self.name = "CombinedABEnhancer"
        
    def enhance_image(self, image_base64):
        """
        A+B 결합: 
        A (사용자 의견): "하얀색 살짝 입힌 느낌"
        B (데이터 분석): 3-4번→5-6번 실제 변화 패턴 구현
        """
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # 메모리 최적화
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
            
            # === B 분석: 3-4번→5-6번 실제 변화 패턴 ===
            
            # 1. 기본 노이즈 제거 (데이터 분석 기반)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.bilateralFilter(cv_image, 7, 25, 25)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 2. B 분석: 밝기 패턴 (베이지 톤→밝은 화이트)
            # 5-6번에서 관찰된 밝기 향상 패턴 적용
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.18)  # 18% 향상 (분석된 값)
            
            # 3. B 분석: 대비 패턴 (깔끔하고 프로페셔널)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.12)  # 12% 향상 (분석된 값)
            
            # 4. B 분석: 색온도 조정 (베이지/핑크→쿨 화이트)
            # LAB 색공간에서 정확한 색온도 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            # A 채널: 베이지 톤 제거 (분석된 패턴)
            a = cv2.add(a, -4)  # 그린 쪽으로 4만큼
            
            # B 채널: 따뜻함 감소 (분석된 패턴)  
            b = cv2.add(b, -6)  # 블루 쪽으로 6만큼
            
            lab_image = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # === A 아이디어: "하얀색 살짝 입힌 느낌" 구현 ===
            
            # 5. A+B 결합: 스마트 화이트 오버레이
            # A의 "하얀색 살짝" 아이디어를 B의 분석으로 정확히 구현
            
            # 5-6번에서 관찰된 정확한 화이트 톤 적용
            smart_white = Image.new('RGB', image.size, (248, 246, 244))  # 분석된 화이트 톤
            image = Image.blend(image, smart_white, 0.07)  # 7% 블렌딩 (분석된 강도)
            
            # 6. B 분석: 선명도 패턴 (웨딩링 디테일 보존)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.15)  # 15% 향상 (분석된 값)
            
            # 7. A+B 최종: 미세 조정
            # A의 "자연스러운 느낌"을 위한 원본과 블렌딩
            original_influence = 0.15  # 15% 원본 보존
            if original_influence > 0:
                original = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
                if max(original.size) > 2048:
                    ratio = 2048 / max(original.size)
                    new_size = tuple(int(dim * ratio) for dim in original.size)
                    original = original.resize(new_size, Image.LANCZOS)
                
                image = Image.blend(image, original, original_influence)
            
            # JPEG 출력
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"A+B Enhancement error: {str(e)}")
            print(traceback.format_exc())
            return None

# 글로벌 enhancer 인스턴스
enhancer = CombinedABEnhancer()

@app.route('/')
def home():
    return jsonify({
        "status": "Wedding Ring Enhancement API - A+B Combined",
        "version": "Combined v1.0",
        "description": "A (user idea: white overlay feeling) + B (data analysis: 3-4→5-6 pattern)",
        "analysis": {
            "user_idea_A": "하얀색 살짝 입힌 느낌",
            "data_pattern_B": "3-4번 원본 → 5-6번 목표 변화 분석",
            "implementation": "A의 직감을 B의 과학적 분석으로 정확히 구현"
        },
        "endpoints": [
            "/health",
            "/enhance_wedding_ring_v6",
            "/enhance_wedding_ring_advanced", 
            "/enhance_wedding_ring_segmented",
            "/enhance_wedding_ring_binary",
            "/enhance_wedding_ring_natural"
        ]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

def enhance_combined():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        image_base64 = data['image_base64']
        
        # A+B 결합 처리
        enhanced_image_bytes = enhancer.enhance_image(image_base64)
        
        if enhanced_image_bytes is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # 바이너리 직접 반환 (Make.com 호환)
        return send_file(
            io.BytesIO(enhanced_image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'ab_combined_{int(datetime.now().timestamp())}.jpg'
        )
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 모든 엔드포인트를 A+B 결합 방식으로 통일
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_v6():
    return enhance_combined()

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_advanced():
    return enhance_combined()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_segmented():
    return enhance_combined()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_binary():
    return enhance_combined()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_natural():
    return enhance_combined()

@app.route('/enhance_wedding_ring_simple', methods=['POST'])
def enhance_simple():
    return enhance_combined()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
