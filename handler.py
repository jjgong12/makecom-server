"""
웨딩링 AI v15.3.4 Perfect Fix - 구문 오류 완전 해결
모든 SyntaxError 제거, 완벽한 Python 구문
"""

import runpod
import sys
import traceback

# 전역 변수로 패키지 상태 관리
PACKAGES_LOADED = False
cv2 = None
np = None
Image = None
ImageEnhance = None
base64 = None
io = None

def safe_import_packages():
    """안전한 패키지 import - 실패해도 크래시하지 않음"""
    global PACKAGES_LOADED, cv2, np, Image, ImageEnhance, base64, io
    
    if PACKAGES_LOADED:
        return True
        
    try:
        import cv2 as cv2_module
        import numpy as np_module  
        from PIL import Image as Image_module
        from PIL import ImageEnhance as ImageEnhance_module
        import base64 as base64_module
        import io as io_module
        
        # 전역 변수에 할당
        cv2 = cv2_module
        np = np_module
        Image = Image_module
        ImageEnhance = ImageEnhance_module
        base64 = base64_module
        io = io_module
        
        PACKAGES_LOADED = True
        return True
        
    except Exception as e:
        print(f"Package import failed: {str(e)}")
        return False

def get_wedding_ring_params():
    """v13.3 파라미터 반환 - 28쌍 학습 데이터 기반"""
    return {
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
                'brightness': 1.12,
                'contrast': 1.05,
                'white_overlay': 0.08,
                'sharpness': 1.12,
                'color_temp_a': 0,
                'color_temp_b': 0,
                'original_blend': 0.22
            },
            'cool': {
                'brightness': 1.18,
                'contrast': 1.12,
                'white_overlay': 0.05,
                'sharpness': 1.18,
                'color_temp_a': 4,
                'color_temp_b': 2,
                'original_blend': 0.18
            }
        },
        'champagne_gold': {
            'natural': {
                'brightness': 1.17,
                'contrast': 1.11,
                'white_overlay': 0.12,
                'sharpness': 1.16,
                'color_temp_a': -4,
                'color_temp_b': -4,
                'original_blend': 0.15
            },
            'warm': {
                'brightness': 1.14,
                'contrast': 1.08,
                'white_overlay': 0.14,
                'sharpness': 1.14,
                'color_temp_a': -6,
                'color_temp_b': -6,
                'original_blend': 0.17
            },
            'cool': {
                'brightness': 1.20,
                'contrast': 1.14,
                'white_overlay': 0.10,
                'sharpness': 1.18,
                'color_temp_a': -2,
                'color_temp_b': -2,
                'original_blend': 0.13
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
                'brightness': 1.13,
                'contrast': 1.06,
                'white_overlay': 0.07,
                'sharpness': 1.12,
                'color_temp_a': 1,
                'color_temp_b': 1,
                'original_blend': 0.24
            },
            'cool': {
                'brightness': 1.19,
                'contrast': 1.12,
                'white_overlay': 0.03,
                'sharpness': 1.16,
                'color_temp_a': 5,
                'color_temp_b': 3,
                'original_blend': 0.20
            }
        }
    }

class PerfectWeddingRingProcessor:
    """완벽한 웨딩링 프로세서 - 구문 오류 없음"""
    
    def __init__(self):
        self.params = None
        
    def detect_black_line_thickness(self, combined_mask, bbox):
        """검은색 선 두께 감지"""
        try:
            if combined_mask is None or bbox is None:
                return 60
                
            x, y, w, h = bbox
            
            if y < 0 or x < 0 or y >= combined_mask.shape[0] or x >= combined_mask.shape[1]:
                return 60
                
            thicknesses = []
            
            safe_y_end = min(y + 50, combined_mask.shape[0])
            safe_x_end = min(x + w, combined_mask.shape[1])
            
            if safe_y_end > y and safe_x_end > x:
                top_line = combined_mask[y:safe_y_end, x:safe_x_end]
                if top_line.size > 0:
                    for col in range(0, min(top_line.shape[1], 20), 5):
                        thickness = 0
                        for row in range(top_line.shape[0]):
                            if row < top_line.shape[0] and col < top_line.shape[1]:
                                if top_line[row, col] > 0:
                                    thickness += 1
                                else:
                                    break
                        if thickness > 10:
                            thicknesses.append(thickness)
            
            if len(thicknesses) > 0:
                return int(np.median(thicknesses))
            else:
                return 60
                
        except Exception as e:
            print(f"두께 감지 실패: {str(e)}")
            return 60
    
    def detect_black_masking(self, image):
        """검은색 마스킹 감지"""
        try:
            if image is None or image.size == 0:
                return None, None, None
                
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            masks = []
            for thresh in [15, 25, 35]:
                try:
                    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                    masks.append(binary)
                except:
                    continue
            
            if not masks:
                return None, None, None
                
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                try:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                except:
                    continue
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, None
            
            best_contour = None
            best_area = 0
            
            for contour in contours:
                try:
                    area = cv2.contourArea(contour)
                    if area < 5000:
                        continue
                        
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4 and area > best_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        ratio = w / h if h > 0 else 0
                        
                        if 0.3 < ratio < 3.0:
                            best_contour = contour
                            best_area = area
                except:
                    continue
            
            if best_contour is None:
                return None, None, None
                
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [best_contour], 255)
            
            bbox = cv2.boundingRect(best_contour)
            
            return mask, best_contour, bbox
            
        except Exception as e:
            print(f"검은색 마스킹 감지 실패: {str(e)}")
            return None, None, None
    
    def detect_metal_type(self, image, mask=None):
        """금속 타입 감지"""
        try:
            if image is None or image.size == 0:
                return 'white_gold'
                
            if mask is not None:
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) == 0:
                    return 'white_gold'
                rgb_values = image[mask_indices[0], mask_indices[1], :]
                if rgb_values.size == 0:
                    return 'white_gold'
                hsv_values = cv2.cvtColor(rgb_values.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                avg_hue = np.mean(hsv_values[:, 0])
                avg_sat = np.mean(hsv_values[:, 1])
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv[:, :, 0])
                avg_sat = np.mean(hsv[:, :, 1])
            
            if avg_sat < 30:
                return 'white_gold'
            elif 5 <= avg_hue <= 25:
                return 'yellow_gold' if avg_sat > 80 else 'champagne_gold'
            elif avg_hue < 5 or avg_hue > 170:
                return 'rose_gold'
            else:
                return 'white_gold'
                
        except Exception as e:
            print(f"금속 타입 감지 실패: {str(e)}")
            return 'white_gold'
    
    def detect_lighting(self, image):
        """조명 감지"""
        try:
            if image is None or image.size == 0:
                return 'natural'
                
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            b_mean = np.mean(lab[:, :, 2])
            
            if b_mean < 125:
                return 'warm'
            elif b_mean > 135:
                return 'cool'
            else:
                return 'natural'
                
        except Exception as e:
            print(f"조명 감지 실패: {str(e)}")
            return 'natural'
    
    def enhance_wedding_ring(self, image, metal_type, lighting):
        """v13.3 웨딩링 보정"""
        try:
            if image is None or image.size == 0:
                return image
                
            if self.params is None:
                self.params = get_wedding_ring_params()
            
            params = self.params.get(metal_type, {}).get(lighting, 
                                   self.params['white_gold']['natural'])
            
            pil_image = Image.fromarray(image)
            
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(params['brightness'])
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(params['contrast'])
            
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(params['sharpness'])
            
            enhanced_array = np.array(enhanced)
            white_overlay = np.full_like(enhanced_array, 255)
            enhanced_array = cv2.addWeighted(
                enhanced_array, 1 - params['white_overlay'],
                white_overlay, params['white_overlay'], 0
            )
            
            lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + params['color_temp_a'], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + params['color_temp_b'], 0, 255)
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            final = cv2.addWeighted(
                enhanced_array, 1 - params['original_blend'],
                image, params['original_blend'], 0
            )
            
            return final.astype(np.uint8)
            
        except Exception as e:
            print(f"웨딩링 보정 실패: {str(e)}")
            return image
    
    def basic_upscale(self, image, scale=2):
        """기본 업스케일링"""
        try:
            if image is None or image.size == 0:
                return image
                
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width * new_height > 50000000:
                scale = 1.5
                new_width = int(width * scale)
                new_height = int(height * scale)
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
        except Exception as e:
            print(f"업스케일링 실패: {str(e)}")
            return image
    
    def inpaint_masking(self, image, mask):
        """인페인팅"""
        try:
            if image is None or mask is None:
                return image
                
            if image.size == 0 or mask.size == 0:
                return image
            
            inpaint_radius = min(5, max(1, int(np.sum(mask > 0) / 10000)))
            
            inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            edge_mask = dilated_mask - mask
            
            if np.any(edge_mask):
                blurred = cv2.GaussianBlur(inpainted, (5, 5), 0)
                result = np.where(edge_mask[..., None] > 0, blurred, inpainted)
            else:
                result = inpainted
                
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"인페인팅 실패: {str(e)}")
            return image
    
    def create_thumbnail(self, image, bbox, target_size=(1000, 1300)):
        """썸네일 생성"""
        try:
            if image is None or bbox is None:
                return image
                
            if image.size == 0:
                return image
                
            x, y, w, h = bbox
            
            margin_w = max(int(w * 0.3), 50)
            margin_h = max(int(h * 0.3), 50)
            
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image.shape[1], x + w + margin_w)
            y2 = min(image.shape[0], y + h + margin_h)
            
            if x2 <= x1 or y2 <= y1:
                return image
            
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return image
            
            target_w, target_h = target_size
            crop_h, crop_w = cropped.shape[:2]
            
            if crop_w == 0 or crop_h == 0:
                return image
            
            ratio_w = target_w / crop_w
            ratio_h = target_h / crop_h
            ratio = min(ratio_w, ratio_h)
            
            new_w = max(1, int(crop_w * ratio))
            new_h = max(1, int(crop_h * ratio))
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            canvas = np.full((target_h, target_w, 3), 240, dtype=np.uint8)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            if start_y >= 0 and start_x >= 0 and start_y + new_h <= target_h and start_x + new_w <= target_w:
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            return canvas
            
        except Exception as e:
            print(f"썸네일 생성 실패: {str(e)}")
            return image

def handler(event):
    """완벽한 핸들러 - 구문 오류 없음"""
    try:
        if not safe_import_packages():
            return {
                "error": "Required packages could not be loaded",
                "status": "package_import_failed"
            }
        
        input_data = event.get("input", {})
        
        if "test" in input_data or not input_data:
            return {
                "success": True,
                "message": "v15.3.4 Perfect Fix - Handler Ready",
                "status": "ready",
                "capabilities": [
                    "적응형 검은색 선 감지",
                    "v13.3 웨딩링 보정",
                    "샴페인골드 화이트화",
                    "안전한 업스케일링",
                    "완벽한 썸네일"
                ],
                "packages_loaded": PACKAGES_LOADED
            }
        
        if "image_base64" not in input_data:
            return {"error": "image_base64 required"}
        
        try:
            image_data = base64.b64decode(input_data["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return {"error": f"이미지 디코딩 실패: {str(e)}"}
        
        processor = PerfectWeddingRingProcessor()
        
        mask, contour, bbox = processor.detect_black_masking(image_array)
        if mask is None:
            return {"error": "검은색 마스킹을 찾을 수 없습니다."}
        
        border_thickness = processor.detect_black_line_thickness(mask, bbox)
        
        x, y, w, h = bbox
        inner_margin = max(border_thickness + 30, 50)
        
        inner_x = max(0, x + inner_margin)
        inner_y = max(0, y + inner_margin)
        inner_w = max(1, w - 2 * inner_margin)
        inner_h = max(1, h - 2 * inner_margin)
        
        if inner_w < 100 or inner_h < 100:
            inner_margin = max(border_thickness // 2, 20)
            inner_x = max(0, x + inner_margin)
            inner_y = max(0, y + inner_margin)
            inner_w = max(100, w - 2 * inner_margin)
            inner_h = max(100, h - 2 * inner_margin)
        
        ring_region = image_array[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
        
        if ring_region.size == 0:
            return {"error": "웨딩링 영역 추출 실패"}
        
        metal_type = processor.detect_metal_type(ring_region)
        lighting = processor.detect_lighting(ring_region)
        
        enhanced_ring = processor.enhance_wedding_ring(ring_region, metal_type, lighting)
        
        upscaled_ring = processor.basic_upscale(enhanced_ring, scale=2)
        
        upscaled_image = processor.basic_upscale(image_array, scale=2)
        
        scale_factor = 2
        scaled_inner_x = inner_x * scale_factor
        scaled_inner_y = inner_y * scale_factor
        scaled_inner_w = upscaled_ring.shape[1]
        scaled_inner_h = upscaled_ring.shape[0]
        
        end_y = min(scaled_inner_y + scaled_inner_h, upscaled_image.shape[0])
        end_x = min(scaled_inner_x + scaled_inner_w, upscaled_image.shape[1])
        
        ring_h = end_y - scaled_inner_y
        ring_w = end_x - scaled_inner_x
        
        if ring_h > 0 and ring_w > 0:
            upscaled_image[scaled_inner_y:end_y, scaled_inner_x:end_x] = upscaled_ring[:ring_h, :ring_w]
        
        upscaled_mask = processor.basic_upscale(mask, scale=2)
        upscaled_mask = np.where(upscaled_mask > 127, 255, 0).astype(np.uint8)
        
        final_image = processor.inpaint_masking(upscaled_image, upscaled_mask)
        
        scaled_bbox = (bbox[0]*2, bbox[1]*2, bbox[2]*2, bbox[3]*2)
        thumbnail = processor.create_thumbnail(final_image, scaled_bbox)
        
        try:
            main_pil = Image.fromarray(final_image)
            main_buffer = io.BytesIO()
            main_pil.save(main_buffer, format='JPEG', quality=95, progressive=True)
            main_base64 = base64.b64encode(main_buffer.getvalue()).decode()
            
            thumb_pil = Image.fromarray(thumbnail)
            thumb_buffer = io.BytesIO()
            thumb_pil.save(thumb_buffer, format='JPEG', quality=95, progressive=True)
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode()
            
            del main_buffer, thumb_buffer, upscaled_image, final_image, thumbnail
            
            return {
                "enhanced_image": main_base64,
                "thumbnail": thumb_base64,
                "processing_info": {
                    "version": "v15.3.4_perfect_fix",
                    "metal_type": metal_type,
                    "lighting": lighting,
                    "border_thickness": border_thickness,
                    "scale_factor": scale_factor,
                    "original_size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                    "final_size": f"{final_image.shape[1]}x{final_image.shape[0]}",
                    "thumbnail_size": "1000x1300",
                    "syntax_error_fixed": True
                }
            }
            
        except Exception as e:
            return {"error": f"이미지 인코딩 실패: {str(e)}"}
    
    except Exception as e:
        error_info = {
            "error": f"전체 처리 실패: {str(e)}",
            "traceback": traceback.format_exc(),
            "version": "v15.3.4_perfect_fix",
            "syntax_error_fixed": True
        }
        return error_info

if __name__ == "__main__":
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"RunPod 시작 실패: {str(e)}")
        sys.exit(1)
