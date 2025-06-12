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
            
            # 2. B 분석: 밝기 패턴 (라이트룸 수준으로 대폭 강화!)
            # 라이트룸 보정 결과 분석: 노출이 상당히 많이 올라감
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.40)  # 40% 향상 (28%→40% 강화)
            
            # 3. B 분석: 대비 패턴 (라이트룸 수준으로 강화!)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.22)  # 22% 향상 (18%→22% 강화)
            
            # 4. B 분석: 색온도 조정 (베이지/핑크→쿨 화이트)
            # LAB 색공간에서 정확한 색온도 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            # A 채널: 베이지 톤 제거 (라이트룸 수준으로 강화!) 
            a = cv2.add(a, -8)  # 그린 쪽으로 8만큼 (6→8 강화)
            
            # B 채널: 따뜻함 감소 (라이트룸의 쿨 톤 구현)  
            b = cv2.add(b, -7)  # 블루 쪽으로 7만큼 (5→7 강화)
            
            lab_image = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # === A 아이디어: "하얀색 살짝 입힌 느낌" 구현 ===
            
            # 5. A+B 결합: 라이트룸 수준 화이트 오버레이 (대폭 강화!)
            # 라이트룸 보정 결과: 배경이 거의 순백색으로 변함
            
            # 순수 화이트로 라이트룸 수준 블렌딩
            pure_white = Image.new('RGB', image.size, (255, 255, 255))  # 완전 순수 화이트
            image = Image.blend(image, pure_white, 0.22)  # 22% 블렌딩 (15%→22% 대폭 강화)
            
            # 6. B 분석: 선명도 패턴 (라이트룸 수준으로 대폭 강화!)
            # 라이트룸 보정: 금속 텍스처와 다이아몬드가 매우 선명해짐
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.30)  # 30% 향상 (20%→30% 강화)
            
            # 7. 라이트룸 수준 하이라이트 부스팅 (금속 반사 강화)
            # 상위 10% 영역을 25% 증가 (라이트룸의 하이라이트 강화 모방)
            img_array = np.array(image)
            highlight_threshold = np.percentile(img_array, 90)  # 상위 10%
            highlight_mask = img_array > highlight_threshold
            img_array[highlight_mask] = np.clip(img_array[highlight_mask] * 1.25, 0, 255)  # 25% 증가
            image = Image.fromarray(img_array.astype(np.uint8))
            
            # 8. A+B 최종: 원본 영향 최소화 (라이트룸 효과 극대화)
            original_influence = 0.02  # 2% 원본 보존 (5%→2% 감소, 라이트룸 효과 극대화)
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
            "user_idea_A": "하얀색 살짝 입힌 느낌 (라이트룸 수준)",
            "data_pattern_B": "라이트룸 보정 전후 실제 변화 분석 적용",
            "implementation": "라이트룸 전문가 수준의 보정을 AI로 재현",
            "adjustments": "라이트룸 분석 기반 모든 파라미터 대폭 강화 (22% 화이트 블렌딩)"
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
