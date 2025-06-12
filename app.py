from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
import base64
import traceback
from datetime import datetime

app = Flask(__name__)

class LightroomLevelEnhancer:
    def __init__(self):
        self.name = "LightroomLevelEnhancer"
        
    def enhance_image(self, image_base64):
        """
        라이트룸 전문가 수준 보정 시스템
        실제 라이트룸 보정 전후 분석 결과를 바탕으로 구현
        """
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # 메모리 최적화 (2K 리샘플링)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
            
            # === 라이트룸 보정 분석 기반 처리 ===
            
            # 1. 기본 노이즈 제거 (전문가 수준)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.bilateralFilter(cv_image, 7, 25, 25)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 2. 라이트룸 수준 밝기 강화 (노출 대폭 증가)
            # 라이트룸 보정 결과 분석: 노출이 상당히 많이 올라감
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.40)  # 40% 향상 (라이트룸 수준)
            
            # 3. 라이트룸 수준 대비 강화 (깔끔하고 프로페셔널)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.22)  # 22% 향상 (라이트룸 수준)
            
            # 4. 라이트룸 수준 색온도 조정 (베이지/핑크→순백색)
            # LAB 색공간에서 정확한 색온도 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            # A 채널: 베이지 톤 제거 (라이트룸 수준으로 강화!) 
            a = cv2.add(a, -8)  # 그린 쪽으로 8만큼 (라이트룸 수준)
            
            # B 채널: 따뜻함 감소 (라이트룸의 쿨 톤 구현)  
            b = cv2.add(b, -7)  # 블루 쪽으로 7만큼 (라이트룸 수준)
            
            lab_image = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 5. 라이트룸 수준 화이트 오버레이 (배경 순백색화)
            # 라이트룸 보정 결과: 배경이 거의 순백색으로 변함
            pure_white = Image.new('RGB', image.size, (255, 255, 255))  # 완전 순수 화이트
            image = Image.blend(image, pure_white, 0.22)  # 22% 블렌딩 (라이트룸 수준)
            
            # 6. 라이트룸 수준 선명도 강화 (금속 텍스처와 다이아몬드)
            # 라이트룸 보정: 금속 텍스처와 다이아몬드가 매우 선명해짐
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.30)  # 30% 향상 (라이트룸 수준)
            
            # 7. 라이트룸 수준 하이라이트 부스팅 (금속 반사 강화)
            # 상위 10% 영역을 25% 증가 (라이트룸의 하이라이트 강화 모방)
            img_array = np.array(image)
            highlight_threshold = np.percentile(img_array, 90)  # 상위 10%
            highlight_mask = img_array > highlight_threshold
            img_array[highlight_mask] = np.clip(img_array[highlight_mask] * 1.25, 0, 255)  # 25% 증가
            image = Image.fromarray(img_array.astype(np.uint8))
            
            # 8. 라이트룸 효과 극대화 (원본 영향 최소화)
            original_influence = 0.02  # 2% 원본 보존 (라이트룸 효과 극대화)
            if original_influence > 0:
                original = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
                if max(original.size) > 2048:
                    ratio = 2048 / max(original.size)
                    new_size = tuple(int(dim * ratio) for dim in original.size)
                    original = original.resize(new_size, Image.LANCZOS)
                
                image = Image.blend(image, original, original_influence)
            
            # 9. 추가 CLAHE 처리 (디테일 극대화)
            cv_final = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_final = cv2.cvtColor(cv_final, cv2.COLOR_BGR2LAB)
            l_final, a_final, b_final = cv2.split(lab_final)
            
            # 적응적 히스토그램 평활화 (라이트룸의 Clarity 효과)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            l_final = clahe.apply(l_final)
            
            lab_final = cv2.merge([l_final, a_final, b_final])
            cv_final = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_final, cv2.COLOR_BGR2RGB))
            
            # 10. 최종 미세 하이라이트 조정 (다이아몬드 반짝임)
            img_array = np.array(image)
            very_bright = img_array > 230  # 매우 밝은 영역 (다이아몬드)
            img_array[very_bright] = np.clip(img_array[very_bright] * 1.08, 0, 255)  # 8% 추가 증가
            image = Image.fromarray(img_array.astype(np.uint8))
            
            # JPEG 출력 (최고 품질)
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"Lightroom-Level Enhancement error: {str(e)}")
            print(traceback.format_exc())
            return None

# 글로벌 enhancer 인스턴스
enhancer = LightroomLevelEnhancer()

@app.route('/')
def home():
    return jsonify({
        "status": "Wedding Ring Enhancement API - Lightroom Level",
        "version": "Lightroom Professional v1.0",
        "description": "Professional-grade enhancement based on actual Lightroom analysis",
        "analysis": {
            "user_idea_A": "하얀색 살짝 입힌 느낌 (라이트룸 수준)",
            "data_pattern_B": "라이트룸 보정 전후 실제 변화 분석 적용",
            "implementation": "라이트룸 전문가 수준의 보정을 AI로 재현",
            "adjustments": "라이트룸 분석 기반 모든 파라미터 대폭 강화 (22% 화이트 블렌딩)"
        },
        "features": [
            "40% 밝기 향상 (라이트룸 노출 수준)",
            "22% 화이트 오버레이 (순백색 배경)",
            "30% 선명도 강화 (금속 텍스처)",
            "하이라이트 부스팅 (다이아몬드 반짝임)",
            "전문가급 색온도 조정",
            "CLAHE 디테일 강화"
        ],
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
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "Lightroom Professional v1.0",
        "enhancement_level": "Professional Studio Quality"
    })

def enhance_lightroom_level():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        image_base64 = data['image_base64']
        
        # 라이트룸 수준 처리
        enhanced_image_bytes = enhancer.enhance_image(image_base64)
        
        if enhanced_image_bytes is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # 바이너리 직접 반환 (Make.com 호환)
        return send_file(
            io.BytesIO(enhanced_image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'lightroom_level_{int(datetime.now().timestamp())}.jpg'
        )
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 모든 엔드포인트를 라이트룸 수준으로 통일
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_v6():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_advanced():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_segmented():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_binary():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_natural():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_simple', methods=['POST'])
def enhance_simple():
    return enhance_lightroom_level()

@app.route('/enhance_wedding_ring_lightroom', methods=['POST'])
def enhance_lightroom():
    return enhance_lightroom_level()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
