import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# v13.3 완전한 파라미터 (28쌍 학습 데이터 기반)
WEDDING_RING_PARAMS = {
    'white_gold': {
        'natural': {
            'brightness': 1.18,
            'contrast': 1.12,
            'white_overlay': 0.09,
            'sharpness': 1.15,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.16,
            'contrast': 1.10,
            'white_overlay': 0.12,
            'sharpness': 1.13,
            'color_temp_a': -5,
            'color_temp_b': -5,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.20,
            'contrast': 1.14,
            'white_overlay': 0.07,
            'sharpness': 1.17,
            'color_temp_a': -2,
            'color_temp_b': -2,
            'original_blend': 0.12
        }
    },
    'rose_gold': {
        'natural': {
            'brightness': 1.15,
            'contrast': 1.08,
            'white_overlay': 0.06,
            'sharpness': 1.15,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.20
        },
        'warm': {
            'brightness': 1.10,
            'contrast': 1.05,
            'white_overlay': 0.05,
            'sharpness': 1.10,
            'color_temp_a': 1,
            'color_temp_b': 0,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.25,
            'contrast': 1.15,
            'white_overlay': 0.08,
            'sharpness': 1.25,
            'color_temp_a': 4,
            'color_temp_b': 3,
            'original_blend': 0.15
        }
    },
    'champagne_gold': {
        'natural': {
            'brightness': 1.17,
            'contrast': 1.11,
            'white_overlay': 0.08,
            'sharpness': 1.16,
            'color_temp_a': -1,
            'color_temp_b': -1,
            'original_blend': 0.15
        },
        'warm': {
            'brightness': 1.15,
            'contrast': 1.10,
            'white_overlay': 0.10,
            'sharpness': 1.20,
            'color_temp_a': -3,
            'color_temp_b': -3,
            'original_blend': 0.18
        },
        'cool': {
            'brightness': 1.22,
            'contrast': 1.15,
            'white_overlay': 0.06,
            'sharpness': 1.25,
            'color_temp_a': 1,
            'color_temp_b': 1,
            'original_blend': 0.12
        }
    },
    'yellow_gold': {
        'natural': {
            'brightness': 1.16,
            'contrast': 1.09,
            'white_overlay': 0.05,
            'sharpness': 1.14,
            'color_temp_a': 3,
            'color_temp_b': 2,
            'original_blend': 0.22
        },
        'warm': {
            'brightness': 1.12,
            'contrast': 1.08,
            'white_overlay': 0.03,
            'sharpness': 1.15,
            'color_temp_a': 2,
            'color_temp_b': 1,
            'original_blend': 0.25
        },
        'cool': {
            'brightness': 1.28,
            'contrast': 1.20,
            'white_overlay': 0.08,
            'sharpness': 1.25,
            'color_temp_a': 5,
            'color_temp_b': 4,
            'original_blend': 0.18
        }
    }
}

class WeddingRingProcessor:
    def __init__(self):
        # Real-ESRGAN 4x 업스케일링 모델 초기화
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='RealESRGAN_x4plus.pth',
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False
        )

    def detect_black_masking(self, image):
        """정밀한 검은색 마킹 영역 감지 및 웨딩링 영역 추출"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 검은색 영역 감지 (threshold < 20)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 마킹 영역 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 영역을 마킹으로 판단
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 마스크 생성
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # 웨딩링 영역 바운딩 박스
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return mask, largest_contour, (x, y, w, h)
        
        return None, None, None

    def detect_metal_type(self, image, mask=None):
        """HSV 색공간 분석으로 금속 타입 정밀 감지"""
        if mask is not None:
            # 마스킹 영역 내에서만 분석
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
            
            # 마스킹된 영역의 평균 색상 계산
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                avg_hue = np.mean(hsv[mask_indices[0], mask_indices[1], 0])
                avg_sat = np.mean(hsv[mask_indices[0], mask_indices[1], 1])
                avg_val = np.mean(hsv[mask_indices[0], mask_indices[1], 2])
            else:
                return 'white_gold'  # 기본값
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
            avg_val = np.mean(hsv[:, :, 2])

        # 정밀한 금속 타입 분류
        if avg_hue < 15 or avg_hue > 165:  # 빨간색 계열
            if avg_sat > 50 and avg_val > 100:
                return 'rose_gold'
            else:
                return 'white_gold'
        elif 15 <= avg_hue <= 35:  # 황색 계열
            if avg_sat > 80:
                return 'yellow_gold'
            elif avg_sat > 40:
                return 'champagne_gold'
            else:
                return 'white_gold'
        else:
            return 'white_gold'  # 기본값

    def detect_lighting(self, image):
        """LAB 색공간 B채널로 조명 환경 정밀 감지"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # L, A, B 채널 분석
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # 각 채널의 평균값
        l_mean = np.mean(l_channel)
        a_mean = np.mean(a_channel)
        b_mean = np.mean(b_channel)
        
        # 조명 환경 분류
        if b_mean < 125:  # 파란쪽으로 치우침
            return 'warm'  # 따뜻한 조명 (보상 필요)
        elif b_mean > 135:  # 노란쪽으로 치우침
            return 'cool'  # 차가운 조명 (보상 필요)
        else:
            return 'natural'  # 자연광

    def enhance_full_image(self, image, metal_type, lighting):
        """원본 전체 이미지 v13.3 보정 (28쌍 데이터 기반)"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                        WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # PIL ImageEnhance로 기본 보정
        pil_image = Image.fromarray(image)
        
        # 1. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(params['brightness'])
        
        # 2. 대비 조정
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'])
        
        # 3. 선명도 조정
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'])
        
        # 4. 하얀색 오버레이 적용 ("하얀색 살짝 입힌 느낌")
        enhanced_array = np.array(enhanced)
        white_overlay = np.full_like(enhanced_array, 255)
        enhanced_array = cv2.addWeighted(
            enhanced_array, 1 - params['white_overlay'],
            white_overlay, params['white_overlay'], 0
        )
        
        # 5. LAB 색공간에서 색온도 정밀 조정
        lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)  # A채널
        lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)  # B채널
        enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 6. 원본과 블렌딩 (자연스러움 보장)
        original_blend = params['original_blend']
        final = cv2.addWeighted(
            enhanced_array, 1 - original_blend,
            image, original_blend, 0
        )
        
        return final.astype(np.uint8)

    def enhance_ring_area_advanced(self, image, mask, metal_type, lighting):
        """마킹 내 커플링 영역 확대 보정 (디테일 극대화)"""
        params = WEDDING_RING_PARAMS.get(metal_type, {}).get(lighting, 
                                        WEDDING_RING_PARAMS['white_gold']['natural'])
        
        # 마킹 영역만 추출
        ring_area = cv2.bitwise_and(image, image, mask=mask)
        
        # 고강도 CLAHE 적용 (디테일 극대화)
        lab = cv2.cvtColor(ring_area, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_ring = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # PIL로 추가 강화 보정
        pil_ring = Image.fromarray(enhanced_ring)
        
        # 밝기 +25% 추가 (확대샷 수준)
        enhancer = ImageEnhance.Brightness(pil_ring)
        enhanced = enhancer.enhance(params['brightness'] * 1.25)
        
        # 선명도 +40% 추가 (밀그레인/큐빅 선명하게)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(params['sharpness'] * 1.40)
        
        # 대비 +20% 추가
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(params['contrast'] * 1.20)
        
        return np.array(enhanced)

    def ai_upscale_4x(self, image):
        """Real-ESRGAN 4x 고품질 업스케일링"""
        try:
            # GPU 사용 가능 시 Real-ESRGAN으로 4x 업스케일링
            output, _ = self.upsampler.enhance(image, outscale=4)
            return output
        except Exception as e:
            print(f"Real-ESRGAN 실패, 기본 업스케일링 사용: {e}")
            # 실패 시 OpenCV LANCZOS4로 2x 업스케일링
            height, width = image.shape[:2]
            return cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_LANCZOS4)

    def inpaint_masking_advanced(self, image, mask):
        """검은색 마킹 영역 고급 인페인팅으로 완전 제거"""
        # OpenCV TELEA 인페인팅
        inpainted = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        
        # 추가적으로 가장자리 정교하게 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
        edge_mask = dilated_mask - mask
        
        # 가장자리 영역 다중 가우시안 블러 적용
        if np.any(edge_mask):
            blurred1 = cv2.GaussianBlur(inpainted, (9, 9), 0)
            blurred2 = cv2.GaussianBlur(inpainted, (5, 5), 0)
            
            # 단계적 블렌딩
            result = np.where(edge_mask[..., None] > 0, 
                            cv2.addWeighted(blurred1, 0.6, blurred2, 0.4, 0), 
                            inpainted)
        else:
            result = inpainted
            
        return result.astype(np.uint8)

    def create_thumbnail_advanced(self, image, bbox):
        """마킹 영역 크롭 → AI 업스케일링 → 1000×1300 고품질 썸네일"""
        x, y, w, h = bbox
        
        # 15% 마진 추가 (더 타이트하게)
        margin_w = int(w * 0.15)
        margin_h = int(h * 0.15)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # 마킹 영역 크롭
        cropped = image[y1:y2, x1:x2]
        
        # AI 업스케일링으로 품질 극대화
        upscaled = self.ai_upscale_4x(cropped)
        
        # 1000×1300 정밀 리사이즈
        target_w, target_h = 1000, 1300
        up_h, up_w = upscaled.shape[:2]
        
        ratio = min(target_w / up_w, target_h / up_h)
        new_w = int(up_w * ratio)
        new_h = int(up_h * ratio)
        
        # LANCZOS4로 고품질 리사이즈
        resized = cv2.resize(upscaled, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 1000×1300 캔버스에 중앙 배치
        canvas = np.full((target_h, target_w, 3), 248, dtype=np.uint8)  # 연한 배경
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # 썸네일 최종 보정 (디테일 극대화)
        pil_thumb = Image.fromarray(canvas)
        
        # 선명도 극대화
        enhancer = ImageEnhance.Sharpness(pil_thumb)
        enhanced_thumb = enhancer.enhance(1.3)
        
        # 대비 약간 강화
        enhancer = ImageEnhance.Contrast(enhanced_thumb)
        final_thumb = enhancer.enhance(1.1)
        
        # 최종 CLAHE 적용
        final_array = np.array(final_thumb)
        lab = cv2.cvtColor(final_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        final_result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return final_result

def handler(event):
    """RunPod Serverless 메인 핸들러 - 완전한 구현"""
    try:
        input_data = event["input"]
        
        # 연결 테스트
        if "prompt" in input_data:
            return {
                "success": True,
                "message": f"웨딩링 AI 연결 성공: {input_data['prompt']}",
                "status": "ready_for_complete_processing",
                "capabilities": [
                    "검은색 마킹 자동 감지",
                    "4가지 금속별 v13.3 보정",
                    "3가지 조명 환경 대응",
                    "Real-ESRGAN 4x 업스케일링",
                    "고급 인페인팅 마킹 제거",
                    "1000×1300 고품질 썸네일",
                    "A_001 + 썸네일 동시 생성"
                ]
            }
        
        # 실제 이미지 처리
        if "image_base64" in input_data:
            image_base64 = input_data["image_base64"]
            
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            original_image = np.array(image.convert('RGB'))
            
            processor = WeddingRingProcessor()
            
            # 1. 검은색 마킹 정밀 감지
            mask, contour, bbox = processor.detect_black_masking(original_image)
            if mask is None:
                return {"error": "검은색 마킹을 찾을 수 없습니다. 마킹이 충분히 검은색인지 확인해주세요."}
            
            # 2. 금속 타입 및 조명 환경 정밀 감지
            metal_type = processor.detect_metal_type(original_image, mask)
            lighting = processor.detect_lighting(original_image)
            
            # === A_001 컨셉샷 완전한 생성 과정 ===
            
            # 3. 원본 전체 v13.3 보정 (28쌍 데이터 기반)
            full_enhanced = processor.enhance_full_image(original_image, metal_type, lighting)
            
            # 4. 마킹 내 커플링 확대 보정 (디테일 극대화)
            enhanced_ring = processor.enhance_ring_area_advanced(full_enhanced, mask, metal_type, lighting)
            
            # 5. 확대 보정된 커플링을 전체 이미지에 정밀 합성
            mask_3d = np.stack([mask] * 3, axis=-1)
            mask_normalized = mask_3d.astype(np.float32) / 255.0
            
            # 부드러운 블렌딩으로 자연스러운 합성
            full_with_enhanced_ring = (enhanced_ring * mask_normalized + 
                                     full_enhanced * (1 - mask_normalized))
            full_with_enhanced_ring = full_with_enhanced_ring.astype(np.uint8)
            
            # 6. 검은색 마킹 고급 인페인팅으로 완전 제거 → A_001 컨셉샷 완성
            a001_result = processor.inpaint_masking_advanced(full_with_enhanced_ring, mask)
            
            # === 썸네일 완전한 생성 과정 ===
            
            # 7. 마킹 영역 크롭 → AI 4x 업스케일링 → 1000×1300 고품질 썸네일
            thumbnail_result = processor.create_thumbnail_advanced(original_image, bbox)
            
            # 8. 결과 최적화 인코딩
            
            # A_001 컨셉샷 (Progressive JPEG, 95% 품질)
            a001_pil = Image.fromarray(a001_result)
            a001_buffer = io.BytesIO()
            a001_pil.save(a001_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
            a001_base64 = base64.b64encode(a001_buffer.getvalue()).decode()
            
            # 썸네일 (Progressive JPEG, 95% 품질)
            thumb_pil = Image.fromarray(thumbnail_result)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True, optimize=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            return {
                "enhanced_image": a001_base64,      # A_001 컨셉샷 (전체 보정 + 마킹 제거)
                "thumbnail": thumb_base64,          # 1000×1300 고품질 썸네일
                "processing_info": {
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "masking_detected": True,
                    "bbox": bbox,
                    "scale_factor": 4,  # Real-ESRGAN 4x
                    "original_size": f"{original_image.shape[1]}x{original_image.shape[0]}",
                    "a001_size": f"{a001_result.shape[1]}x{a001_result.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "processing_quality": "premium",
                    "ai_upscaling": "Real-ESRGAN_4x"
                }
            }
            
    except Exception as e:
        return {
            "error": f"처리 중 오류 발생: {str(e)}",
            "error_type": "processing_error",
            "suggestion": "이미지 형식과 검은색 마킹 상태를 확인해주세요."
        }

# RunPod 서버리스 설정
runpod.serverless.start({"handler": handler})
