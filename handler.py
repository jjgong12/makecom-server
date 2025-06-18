import runpod
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import io
import base64
from typing import Optional, Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeddingRingEnhancerV23:
    def __init__(self):
        """v23.0 완전한 웨딩링 처리 시스템 - 검은테두리 완전제거 + 색상왜곡 해결"""
        
        # v13.3 완전한 4금속×3조명 = 12가지 파라미터 (28쌍 학습 데이터 기반)
        self.params = {
            'white_gold': {
                'natural': {
                    'brightness': 1.28, 'contrast': 1.18, 'sharpness': 1.35, 'clarity': 1.20,
                    'saturation': 1.08, 'warmth': 0.95, 'white_overlay': 0.12,
                    'gamma': 1.15, 'color_temp_a': -2, 'color_temp_b': -2
                },
                'warm': {
                    'brightness': 1.32, 'contrast': 1.22, 'sharpness': 1.40, 'clarity': 1.25,
                    'saturation': 1.12, 'warmth': 1.05, 'white_overlay': 0.08,
                    'gamma': 1.12, 'color_temp_a': -1, 'color_temp_b': -1
                },
                'cool': {
                    'brightness': 1.25, 'contrast': 1.15, 'sharpness': 1.30, 'clarity': 1.18,
                    'saturation': 1.05, 'warmth': 0.88, 'white_overlay': 0.15,
                    'gamma': 1.18, 'color_temp_a': -3, 'color_temp_b': -3
                }
            },
            'yellow_gold': {
                'natural': {
                    'brightness': 1.25, 'contrast': 1.20, 'sharpness': 1.25, 'clarity': 1.15,
                    'saturation': 1.15, 'warmth': 1.08, 'white_overlay': 0.05,
                    'gamma': 1.10, 'color_temp_a': 1, 'color_temp_b': 2
                },
                'warm': {
                    'brightness': 1.30, 'contrast': 1.25, 'sharpness': 1.30, 'clarity': 1.20,
                    'saturation': 1.20, 'warmth': 1.12, 'white_overlay': 0.03,
                    'gamma': 1.08, 'color_temp_a': 2, 'color_temp_b': 3
                },
                'cool': {
                    'brightness': 1.22, 'contrast': 1.18, 'sharpness': 1.22, 'clarity': 1.12,
                    'saturation': 1.10, 'warmth': 1.02, 'white_overlay': 0.08,
                    'gamma': 1.12, 'color_temp_a': 0, 'color_temp_b': 1
                }
            },
            'rose_gold': {
                'natural': {
                    'brightness': 1.26, 'contrast': 1.19, 'sharpness': 1.28, 'clarity': 1.17,
                    'saturation': 1.12, 'warmth': 1.06, 'white_overlay': 0.06,
                    'gamma': 1.11, 'color_temp_a': 1, 'color_temp_b': -1
                },
                'warm': {
                    'brightness': 1.29, 'contrast': 1.23, 'sharpness': 1.32, 'clarity': 1.21,
                    'saturation': 1.16, 'warmth': 1.10, 'white_overlay': 0.04,
                    'gamma': 1.09, 'color_temp_a': 2, 'color_temp_b': 0
                },
                'cool': {
                    'brightness': 1.24, 'contrast': 1.16, 'sharpness': 1.25, 'clarity': 1.14,
                    'saturation': 1.08, 'warmth': 0.98, 'white_overlay': 0.09,
                    'gamma': 1.13, 'color_temp_a': 0, 'color_temp_b': -2
                }
            },
            'champagne_gold': {
                'natural': {
                    'brightness': 1.30, 'contrast': 1.21, 'sharpness': 1.33, 'clarity': 1.22,
                    'saturation': 1.02, 'warmth': 1.04, 'white_overlay': 0.15,
                    'gamma': 1.14, 'color_temp_a': -6, 'color_temp_b': -6
                },
                'warm': {
                    'brightness': 1.33, 'contrast': 1.25, 'sharpness': 1.37, 'clarity': 1.26,
                    'saturation': 1.05, 'warmth': 1.08, 'white_overlay': 0.12,
                    'gamma': 1.11, 'color_temp_a': -5, 'color_temp_b': -5
                },
                'cool': {
                    'brightness': 1.27, 'contrast': 1.18, 'sharpness': 1.30, 'clarity': 1.19,
                    'saturation': 0.98, 'warmth': 0.96, 'white_overlay': 0.18,
                    'gamma': 1.16, 'color_temp_a': -7, 'color_temp_b': -7
                }
            }
        }
        
        # 28쌍 AFTER 파일 학습 기반 배경색 시스템
        self.after_bg_colors = {
            'white_gold': {
                'natural': [252, 250, 248], 'warm': [254, 252, 250], 'cool': [250, 248, 245]
            },
            'yellow_gold': {
                'natural': [251, 249, 246], 'warm': [253, 251, 248], 'cool': [249, 247, 244]
            },
            'rose_gold': {
                'natural': [252, 248, 246], 'warm': [254, 250, 248], 'cool': [250, 246, 244]
            },
            'champagne_gold': {
                'natural': [253, 251, 249], 'warm': [255, 253, 251], 'cool': [251, 249, 247]
            }
        }

    def detect_and_remove_black_border_v23(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """v23.0 핵심: 40% 스캔 + 중앙영역 체크로 검은테두리 완전제거"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # v23.0 핵심 변경: 최대 40% 영역까지 스캔 (기존 200픽셀 → 40%)
        max_border = min(int(h * 0.4), int(w * 0.4), 400)
        
        borders = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        found_border = False
        
        # 상단 검은 테두리 감지 (중앙 50% 영역만 체크)
        for y in range(max_border):
            row_mean = np.mean(gray[y, w//4:3*w//4])  # 중앙 50% 영역만
            if row_mean < 80:
                borders['top'] = y + 1
                found_border = True
            else:
                break
        
        # 하단 검은 테두리 감지
        for y in range(h-1, h-max_border-1, -1):
            row_mean = np.mean(gray[y, w//4:3*w//4])
            if row_mean < 80:
                borders['bottom'] = h - y
                found_border = True
            else:
                break
        
        # 좌측 검은 테두리 감지
        for x in range(max_border):
            col_mean = np.mean(gray[h//4:3*h//4, x])  # 중앙 50% 영역만
            if col_mean < 80:
                borders['left'] = x + 1
                found_border = True
            else:
                break
        
        # 우측 검은 테두리 감지
        for x in range(w-1, w-max_border-1, -1):
            col_mean = np.mean(gray[h//4:3*h//4, x])
            if col_mean < 80:
                borders['right'] = w - x
                found_border = True
            else:
                break
        
        # 검은 테두리가 발견되면 크롭
        if found_border:
            y1 = borders['top']
            y2 = h - borders['bottom']
            x1 = borders['left']
            x2 = w - borders['right']
            
            # 안전 마진 추가
            safety_margin = 5
            y1 = max(0, y1 - safety_margin)
            y2 = min(h, y2 + safety_margin)
            x1 = max(0, x1 - safety_margin)
            x2 = min(w, x2 + safety_margin)
            
            cropped = image[y1:y2, x1:x2]
            
            # 2차 정밀 크롭 (v23.0 추가)
            if cropped.shape[0] > 20 and cropped.shape[1] > 20:
                gray2 = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                if np.mean(gray2[:10, :]) < 100:  # 상단이 여전히 어두우면
                    edge_cut = 15
                    cropped = cropped[edge_cut:, :]
                if np.mean(gray2[-10:, :]) < 100:  # 하단이 여전히 어두우면
                    edge_cut = 15
                    cropped = cropped[:-edge_cut, :]
                if np.mean(gray2[:, :10]) < 100:  # 좌측이 여전히 어두우면
                    edge_cut = 15
                    cropped = cropped[:, edge_cut:]
                if np.mean(gray2[:, -10:]) < 100:  # 우측이 여전히 어두우면
                    edge_cut = 15
                    cropped = cropped[:, :-edge_cut]
            
            logger.info(f"v23.0 Black border removed: {borders}")
            return cropped, True
        
        return image, False

    def detect_metal_type(self, image: np.ndarray) -> str:
        """간단한 RGB 기반 금속 타입 감지 (HSV 변환 제거)"""
        try:
            # v23.0 핵심: HSV 변환 제거, RGB 기반으로 단순화
            mean_r = np.mean(image[:, :, 0])
            mean_g = np.mean(image[:, :, 1])
            mean_b = np.mean(image[:, :, 2])
            
            # RGB 비율로 금속 타입 분류
            if mean_r > mean_g + 10 and mean_r > mean_b + 10:
                if mean_r - mean_g > 20:
                    return 'rose_gold'
                else:
                    return 'yellow_gold'
            elif mean_r < 180 and mean_g < 180 and mean_b < 180:
                return 'champagne_gold'
            else:
                return 'white_gold'
                
        except Exception as e:
            logger.warning(f"Metal detection failed: {e}")
            return 'white_gold'

    def detect_lighting(self, image: np.ndarray) -> str:
        """간단한 밝기 기반 조명 감지 (LAB 변환 제거)"""
        try:
            # v23.0 핵심: LAB 변환 제거, 단순 밝기 기반
            brightness = np.mean(image)
            
            if brightness < 120:
                return 'warm'
            elif brightness > 160:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            logger.warning(f"Lighting detection failed: {e}")
            return 'natural'

    def enhance_wedding_ring_v23(self, image: np.ndarray, metal_type: str, lighting: str) -> np.ndarray:
        """v23.0 안전한 PIL 기반 보정 (HSV/LAB 변환 완전 제거)"""
        try:
            params = self.params.get(metal_type, {}).get(lighting, self.params['white_gold']['natural'])
            
            # PIL로 변환
            pil_image = Image.fromarray(image)
            
            # 1. 노이즈 제거 (가벼운 블러)
            enhanced = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # 2. 밝기 조정 (PIL 기반 - 안전)
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(params['brightness'])
            
            # 3. 대비 조정 (PIL 기반 - 안전)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            # 4. 선명도 조정 (PIL 기반 - 안전)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            # v23.0 핵심: HSV/LAB 색공간 변환 완전 제거!
            # (색상 왜곡의 주범이었음)
            
            # NumPy로 변환
            enhanced = np.array(enhanced)
            
            # 5. 미묘한 화이트 오버레이 (샴페인골드용)
            if metal_type == 'champagne_gold':
                white_strength = params.get('white_overlay', 0.1)
                white_overlay = np.full_like(enhanced, 255, dtype=np.uint8)
                enhanced = cv2.addWeighted(enhanced, 1-white_strength, white_overlay, white_strength, 0)
            
            # 6. 부드러운 블렌딩 (80% 보정 + 20% 원본)
            enhanced = cv2.addWeighted(enhanced, 0.8, image, 0.2, 0)
            
            return enhanced.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image

    def create_white_background_v23(self, image: np.ndarray, metal_type: str, lighting: str) -> np.ndarray:
        """v23.0 완전 흰색 배경 생성 (248 RGB)"""
        try:
            h, w = image.shape[:2]
            
            # v23.0 핵심: 배경을 248 (거의 흰색)으로 교체
            background = np.full_like(image, 248, dtype=np.uint8)
            
            # 간단한 마스크 생성 (웨딩링 영역 보호)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # 형태학적 연산으로 마스크 정제
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 가우시안 블러로 부드러운 경계
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask_norm = mask.astype(np.float32) / 255.0
            
            # 3채널로 확장
            mask_3ch = np.stack([mask_norm] * 3, axis=-1)
            
            # 배경 합성
            result = background.astype(np.float32) * (1 - mask_3ch) + image.astype(np.float32) * mask_3ch
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Background creation failed: {e}")
            return image

    def create_perfect_thumbnail_v23(self, image: np.ndarray) -> np.ndarray:
        """v23.0 썸네일 90% 확대 (기존 80% → 90%)"""
        try:
            target_width, target_height = 1000, 1300
            h, w = image.shape[:2]
            
            # 현재 비율 계산
            current_ratio = w / h
            target_ratio = target_width / target_height
            
            if current_ratio > target_ratio:
                # 이미지가 더 넓음 - 높이 기준으로 맞춤
                new_height = target_height
                new_width = int(target_height * current_ratio)
            else:
                # 이미지가 더 높음 - 너비 기준으로 맞춤
                new_width = target_width
                new_height = int(target_width / current_ratio)
            
            # 리사이즈
            pil_image = Image.fromarray(image)
            resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 흰색 캔버스 생성
            canvas = Image.new('RGB', (target_width, target_height), color=(248, 248, 248))
            
            # 중앙 배치 (90% 확대를 위한 조정)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # v23.0: 90% 확대 효과 (10% 확대 적용)
            scale_factor = 1.125  # 12.5% 확대 (기존보다 더 크게)
            final_width = int(new_width * scale_factor)
            final_height = int(new_height * scale_factor)
            
            if final_width <= target_width and final_height <= target_height:
                resized = resized.resize((final_width, final_height), Image.Resampling.LANCZOS)
                paste_x = (target_width - final_width) // 2
                paste_y = (target_height - final_height) // 2
            
            canvas.paste(resized, (paste_x, paste_y))
            
            return np.array(canvas)
            
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            return image

    def process_image(self, image_data: str, output_format: str = "enhanced") -> Dict:
        """v23.0 메인 처리 함수"""
        try:
            # Base64 디코딩
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image.convert('RGB'))
            
            results = {}
            
            # 1. v23.0 검은 테두리 제거
            processed_image, border_removed = self.detect_and_remove_black_border_v23(image_np)
            logger.info(f"Border removal: {border_removed}")
            
            # 2. 금속 타입 및 조명 감지 (HSV/LAB 변환 제거)
            metal_type = self.detect_metal_type(processed_image)
            lighting = self.detect_lighting(processed_image)
            logger.info(f"Detected: {metal_type}, {lighting}")
            
            # 3. v23.0 안전한 웨딩링 보정 (색상 왜곡 방지)
            enhanced_image = self.enhance_wedding_ring_v23(processed_image, metal_type, lighting)
            
            # 4. v23.0 완전 흰색 배경 생성
            final_image = self.create_white_background_v23(enhanced_image, metal_type, lighting)
            
            # 5. v23.0 썸네일 90% 확대
            thumbnail = self.create_perfect_thumbnail_v23(final_image)
            
            # 결과 저장
            if output_format == "enhanced":
                results['enhanced'] = self._image_to_base64(final_image)
            elif output_format == "thumbnail":
                results['thumbnail'] = self._image_to_base64(thumbnail)
            elif output_format == "both":
                results['enhanced'] = self._image_to_base64(final_image)
                results['thumbnail'] = self._image_to_base64(thumbnail)
            
            results.update({
                'metal_type': metal_type,
                'lighting': lighting,
                'border_removed': border_removed,
                'version': 'v23.0_final'
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'error': str(e)}

    def _image_to_base64(self, image_np: np.ndarray) -> str:
        """NumPy 배열을 Base64로 변환"""
        pil_image = Image.fromarray(image_np)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

# RunPod Handler
def handler(event):
    """RunPod 핸들러 함수"""
    try:
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        output_format = input_data.get('format', 'both')
        
        if not image_data:
            return {'error': 'No image data provided'}
        
        # v23.0 프로세서 초기화
        processor = WeddingRingEnhancerV23()
        
        # 이미지 처리
        results = processor.process_image(image_data, output_format)
        
        # RunPod Output 중첩 구조 필수
        return {
            'output': results
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'error': str(e)
        }

# RunPod 서버리스 시작
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
