from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
import base64
import traceback
from datetime import datetime

app = Flask(__name__)

class BalancedMidVersionEnhancer:
    def __init__(self):
        self.name = "BalancedMidVersionEnhancer"
        
    def enhance_image(self, image_base64):
        """
        버전10 + 버전17 중간값 시스템 (버전13.5)
        안정성과 개선 효과의 완벽한 균형
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
            
            # === 버전10 기준 + 라이트룸 살짝 반영 ===
            
            # 1. 기본 노이즈 제거 (약하게)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.bilateralFilter(cv_image, 5, 20, 20)  # 약하게 처리
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 2. 밝기 조정 (버전10과 버전17의 중간)
            # 버전10: 1.18, 버전17: 1.22 → 중간: 1.20
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.20)  # 20% 향상 (중간값)
            
            # 3. 대비 조정 (버전10과 버전17의 중간)
            # 버전10: 1.12, 버전17: 1.15 → 중간: 1.14
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.14)  # 14% 향상 (중간값)
            
            # 4. 색온도 조정 (매우 약하게, 버전10 기준)
            # LAB 색공간에서 살짝만 조정
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            # A 채널: 베이지 톤 제거 (중간값)
            # 버전10: -4, 버전17: -3 → 중간: -3.5 ≈ -4
            a = cv2.add(a, -4)  # 4만큼 (중간값)
            
            # B 채널: 따뜻함 감소 (중간값)  
            # 버전10: -3, 버전17: -4 → 중간: -3.5 ≈ -4
            b = cv2.add(b, -4)  # 4만큼 (중간값)
            
            lab_image = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 5. 화이트 오버레이 (버전10과 버전17의 중간)
            # 버전10: 10%, 버전17: 12% → 중간: 11%
            pure_white = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.blend(image, pure_white, 0.11)  # 11% 블렌딩 (중간값)
            
            # 6. 선명도 조정 (버전10과 버전17의 중간)
            # 버전10: 1.15, 버전17: 1.18 → 중간: 1.17
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.17)  # 17% 향상 (중간값)
            
            # 7. 미세한 하이라이트 부스팅 (매우 약하게)
            # 상위 5% 영역만 8% 증가 (매우 보수적)
            img_array = np.array(image)
            highlight_threshold = np.percentile(img_array, 95)  # 상위 5%만
            highlight_mask = img_array > highlight_threshold
            img_array[highlight_mask] = np.clip(img_array[highlight_mask] * 1.08, 0, 255)  # 8% 증가
            image = Image.fromarray(img_array.astype(np.uint8))
            
            # 8. 원본과 블렌딩 (중간값)
            # 버전10: 15%, 버전17: 10% → 중간: 12.5% ≈ 12%
            original_influence = 0.12  # 12% 원본 보존 (중간값)
            if original_influence > 0:
                original = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert('RGB')
                if max(original.size) > 2048:
                    ratio = 2048 / max(original.size)
                    new_size = tuple(int(dim * ratio) for dim in original.size)
                    original = original.resize(new_size, Image.LANCZOS)
                
                image = Image.blend(image, original, original_influence)
            
            # 9. 매우 약한 CLAHE (디테일 살짝만)
            cv_final = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_final = cv2.cvtColor(cv_final, cv2.COLOR_BGR2LAB)
            l_final, a_final, b_final = cv2.split(lab_final)
            
            # 매우 약한 적응적 히스토그램 평활화
            clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(16,16))  # 매우 약하게
            l_final = clahe.apply(l_final)
            
            lab_final = cv2.merge([l_final, a_final, b_final])
            cv_final = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_final, cv2.COLOR_BGR2RGB))
            
            # JPEG 출력
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"Balanced Enhancement error: {str(e)}")
            print(traceback.format_exc())
            return None

# 글로벌 enhancer 인스턴스
enhancer = BalancedMidVersionEnhancer()

@app.route('/')
def home():
    return jsonify({
        "status": "Wedding Ring Enhancement API - Perfect Balance",
        "version": "Balanced v13.5",
        "description": "Perfect balance between Version 10 stability and Version 17 enhancement",
        "analysis": {
            "base": "Version 10 (proven stable) + Version 17 (enhanced) 중간값",
            "philosophy": "안정성과 효과의 완벽한 균형",
            "method": "수학적 중간값 계산으로 최적 파라미터 도출",
            "result": "자연스러우면서도 확실한 개선 효과"
        },
        "parameters": {
            "brightness": "1.20 (20% - 중간값)",
            "contrast": "1.14 (14% - 중간값)",
            "white_overlay": "11% (중간값)",
            "sharpness": "1.17 (17% - 중간값)",
            "color_temp": "A:-4, B:-4 (중간값)",
            "original_blend": "12% (중간값)"
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
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "Balanced v13.5",
        "enhancement_level": "Perfect Balance - Stable & Effective"
    })

def enhance_balanced():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        image_base64 = data['image_base64']
        
        # 균형잡힌 처리
        enhanced_image_bytes = enhancer.enhance_image(image_base64)
        
        if enhanced_image_bytes is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # 바이너리 직접 반환 (Make.com 호환)
        return send_file(
            io.BytesIO(enhanced_image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'balanced_v13_5_{int(datetime.now().timestamp())}.jpg'
        )
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 모든 엔드포인트를 균형잡힌 처리로 통일
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_v6():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_advanced():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_segmented():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_binary():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_natural():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_simple', methods=['POST'])
def enhance_simple():
    return enhance_balanced()

@app.route('/enhance_wedding_ring_balanced', methods=['POST'])
def enhance_balanced_route():
    return enhance_balanced()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
