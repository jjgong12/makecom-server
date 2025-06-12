from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import io
import base64
import traceback
from datetime import datetime

app = Flask(__name__)

class GentleLightroomEnhancer:
    def __init__(self):
        self.name = "GentleLightroomEnhancer"
        
    def enhance_image(self, image_base64):
        """
        버전10 + 라이트룸 살짝 반영 시스템
        자연스럽게 조금만 개선
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
            
            # 2. 밝기 조정 (버전10 기준 + 라이트룸 살짝)
            # 1.18 → 1.22 (4% 증가만)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.22)  # 22% 향상 (자연스럽게)
            
            # 3. 대비 조정 (버전10 기준 + 라이트룸 살짝)
            # 1.12 → 1.15 (3% 증가만)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)  # 15% 향상 (자연스럽게)
            
            # 4. 색온도 조정 (매우 약하게, 버전10 기준)
            # LAB 색공간에서 살짝만 조정
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            
            # A 채널: 베이지 톤 제거 (매우 약하게)
            a = cv2.add(a, -3)  # 3만큼만 (자연스럽게)
            
            # B 채널: 따뜻함 감소 (매우 약하게)  
            b = cv2.add(b, -4)  # 4만큼만 (자연스럽게)
            
            lab_image = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 5. 화이트 오버레이 (버전10 기준 + 살짝)
            # 10% → 12% (2% 증가만)
            pure_white = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.blend(image, pure_white, 0.12)  # 12% 블렌딩 (자연스럽게)
            
            # 6. 선명도 조정 (버전10 기준 + 살짝)
            # 1.15 → 1.18 (3% 증가만)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.18)  # 18% 향상 (자연스럽게)
            
            # 7. 미세한 하이라이트 부스팅 (매우 약하게)
            # 상위 5% 영역만 8% 증가 (매우 보수적)
            img_array = np.array(image)
            highlight_threshold = np.percentile(img_array, 95)  # 상위 5%만
            highlight_mask = img_array > highlight_threshold
            img_array[highlight_mask] = np.clip(img_array[highlight_mask] * 1.08, 0, 255)  # 8% 증가
            image = Image.fromarray(img_array.astype(np.uint8))
            
            # 8. 원본과 블렌딩 (자연스러움 유지)
            # 원본 영향 10% (자연스러움 보장)
            original_influence = 0.10  # 10% 원본 보존
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
            print(f"Gentle Enhancement error: {str(e)}")
            print(traceback.format_exc())
            return None

# 글로벌 enhancer 인스턴스
enhancer = GentleLightroomEnhancer()

@app.route('/')
def home():
    return jsonify({
        "status": "Wedding Ring Enhancement API - Gentle Version",
        "version": "Gentle v10.5",
        "description": "Natural enhancement based on Version 10 + gentle Lightroom touches",
        "analysis": {
            "base": "Version 10 parameters (proven stable)",
            "enhancement": "Gentle Lightroom analysis application",
            "philosophy": "자연스러움 우선, 과도한 보정 방지",
            "adjustments": "버전10 + 라이트룸 살짝 (12% 화이트 블렌딩)"
        },
        "parameters": {
            "brightness": "1.22 (22% - 자연스럽게)",
            "contrast": "1.15 (15% - 균형있게)",
            "white_overlay": "12% (자연스럽게)",
            "sharpness": "1.18 (18% - 적절하게)",
            "color_temp": "매우 약한 조정",
            "original_blend": "10% (자연스러움 보장)"
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
        "version": "Gentle v10.5",
        "enhancement_level": "Natural & Balanced"
    })

def enhance_gentle():
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"error": "No image_base64 provided"}), 400
        
        image_base64 = data['image_base64']
        
        # 자연스러운 처리
        enhanced_image_bytes = enhancer.enhance_image(image_base64)
        
        if enhanced_image_bytes is None:
            return jsonify({"error": "Enhancement failed"}), 500
        
        # 바이너리 직접 반환 (Make.com 호환)
        return send_file(
            io.BytesIO(enhanced_image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'gentle_enhanced_{int(datetime.now().timestamp())}.jpg'
        )
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 모든 엔드포인트를 자연스러운 처리로 통일
@app.route('/enhance_wedding_ring_v6', methods=['POST'])
def enhance_v6():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_advanced', methods=['POST'])
def enhance_advanced():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_segmented', methods=['POST'])
def enhance_segmented():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_binary', methods=['POST'])
def enhance_binary():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_natural', methods=['POST'])
def enhance_natural():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_simple', methods=['POST'])
def enhance_simple():
    return enhance_gentle()

@app.route('/enhance_wedding_ring_gentle', methods=['POST'])
def enhance_gentle_route():
    return enhance_gentle()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
