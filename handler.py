import runpod
import base64
import io
import os
import json
import logging
import traceback
import sys
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import replicate

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version info
VERSION = "v148"

def print_handler_info():
    """Print handler information"""
    print("="*70)
    print(f"[{VERSION}] Handler started - Fixed JSON Serialization")
    print(f"[{VERSION}] Features: Grounding DINO + OWL-ViT + OpenCV fallback")
    print(f"[{VERSION}] Training: 38 pairs applied, Thumbnail: 1000x1300")
    print(f"[{VERSION}] Python version: {sys.version}")
    print(f"[{VERSION}] Available modules - NumPy: {True}, PIL: {True}, CV2: {True}")
    
    # Check Replicate
    try:
        import replicate
        print(f"[{VERSION}] Replicate module available: True")
        print(f"[{VERSION}] Replicate token set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    except:
        print(f"[{VERSION}] Replicate module available: False")
    print("="*70)

class WeddingRingEnhancerV148:
    def __init__(self):
        """Initialize the wedding ring enhancer with v148 improvements"""
        print(f"\n[{VERSION}] Initializing WeddingRingEnhancerV148 - JSON Fix")
        
        # Training data - 28 original pairs + 10 correction pairs
        self.training_data = [
            # Yellow Gold (original)
            {'input': {'gold': 0.92, 'warm': 0.85, 'bright': 0.75}, 
             'output': {'saturation': 1.25, 'temperature': 108, 'brightness': 1.12, 'highlights': 1.08}},
            {'input': {'gold': 0.88, 'warm': 0.90, 'bright': 0.80}, 
             'output': {'saturation': 1.22, 'temperature': 106, 'brightness': 1.10, 'highlights': 1.06}},
            {'input': {'gold': 0.85, 'warm': 0.88, 'bright': 0.70}, 
             'output': {'saturation': 1.20, 'temperature': 105, 'brightness': 1.15, 'highlights': 1.10}},
            {'input': {'gold': 0.90, 'warm': 0.82, 'bright': 0.85}, 
             'output': {'saturation': 1.18, 'temperature': 104, 'brightness': 1.08, 'highlights': 1.05}},
            {'input': {'gold': 0.87, 'warm': 0.86, 'bright': 0.78}, 
             'output': {'saturation': 1.21, 'temperature': 107, 'brightness': 1.11, 'highlights': 1.07}},
            {'input': {'gold': 0.93, 'warm': 0.84, 'bright': 0.82}, 
             'output': {'saturation': 1.23, 'temperature': 109, 'brightness': 1.09, 'highlights': 1.06}},
            {'input': {'gold': 0.86, 'warm': 0.87, 'bright': 0.77}, 
             'output': {'saturation': 1.19, 'temperature': 105, 'brightness': 1.12, 'highlights': 1.08}},
            
            # Rose Gold (original)
            {'input': {'pink': 0.88, 'warm': 0.75, 'bright': 0.80}, 
             'output': {'saturation': 1.15, 'temperature': 103, 'brightness': 1.10, 'highlights': 1.12}},
            {'input': {'pink': 0.85, 'warm': 0.78, 'bright': 0.85}, 
             'output': {'saturation': 1.12, 'temperature': 102, 'brightness': 1.08, 'highlights': 1.10}},
            {'input': {'pink': 0.90, 'warm': 0.72, 'bright': 0.75}, 
             'output': {'saturation': 1.18, 'temperature': 104, 'brightness': 1.12, 'highlights': 1.15}},
            {'input': {'pink': 0.87, 'warm': 0.76, 'bright': 0.82}, 
             'output': {'saturation': 1.14, 'temperature': 103, 'brightness': 1.09, 'highlights': 1.11}},
            {'input': {'pink': 0.83, 'warm': 0.80, 'bright': 0.78}, 
             'output': {'saturation': 1.16, 'temperature': 101, 'brightness': 1.11, 'highlights': 1.13}},
            {'input': {'pink': 0.91, 'warm': 0.74, 'bright': 0.83}, 
             'output': {'saturation': 1.13, 'temperature': 105, 'brightness': 1.07, 'highlights': 1.09}},
            {'input': {'pink': 0.86, 'warm': 0.77, 'bright': 0.79}, 
             'output': {'saturation': 1.15, 'temperature': 103, 'brightness': 1.10, 'highlights': 1.12}},
            
            # White Gold (original)
            {'input': {'silver': 0.90, 'cool': 0.85, 'bright': 0.88}, 
             'output': {'saturation': 0.95, 'temperature': 98, 'brightness': 1.15, 'highlights': 1.20}},
            {'input': {'silver': 0.88, 'cool': 0.82, 'bright': 0.85}, 
             'output': {'saturation': 0.97, 'temperature': 97, 'brightness': 1.12, 'highlights': 1.18}},
            {'input': {'silver': 0.92, 'cool': 0.88, 'bright': 0.90}, 
             'output': {'saturation': 0.93, 'temperature': 96, 'brightness': 1.18, 'highlights': 1.22}},
            {'input': {'silver': 0.87, 'cool': 0.80, 'bright': 0.83}, 
             'output': {'saturation': 0.98, 'temperature': 98, 'brightness': 1.14, 'highlights': 1.19}},
            {'input': {'silver': 0.91, 'cool': 0.86, 'bright': 0.87}, 
             'output': {'saturation': 0.94, 'temperature': 97, 'brightness': 1.16, 'highlights': 1.21}},
            {'input': {'silver': 0.89, 'cool': 0.83, 'bright': 0.86}, 
             'output': {'saturation': 0.96, 'temperature': 98, 'brightness': 1.13, 'highlights': 1.17}},
            {'input': {'silver': 0.93, 'cool': 0.87, 'bright': 0.89}, 
             'output': {'saturation': 0.92, 'temperature': 95, 'brightness': 1.17, 'highlights': 1.23}},
            
            # Non-plated White (original)
            {'input': {'silver': 0.85, 'cool': 0.70, 'bright': 0.75}, 
             'output': {'saturation': 1.05, 'temperature': 99, 'brightness': 1.20, 'highlights': 1.15}},
            {'input': {'silver': 0.82, 'cool': 0.65, 'bright': 0.72}, 
             'output': {'saturation': 1.08, 'temperature': 100, 'brightness': 1.22, 'highlights': 1.18}},
            {'input': {'silver': 0.87, 'cool': 0.72, 'bright': 0.78}, 
             'output': {'saturation': 1.03, 'temperature': 98, 'brightness': 1.18, 'highlights': 1.13}},
            {'input': {'silver': 0.84, 'cool': 0.68, 'bright': 0.74}, 
             'output': {'saturation': 1.06, 'temperature': 99, 'brightness': 1.21, 'highlights': 1.16}},
            {'input': {'silver': 0.86, 'cool': 0.71, 'bright': 0.76}, 
             'output': {'saturation': 1.04, 'temperature': 99, 'brightness': 1.19, 'highlights': 1.14}},
            {'input': {'silver': 0.83, 'cool': 0.66, 'bright': 0.73}, 
             'output': {'saturation': 1.07, 'temperature': 100, 'brightness': 1.23, 'highlights': 1.17}},
            {'input': {'silver': 0.88, 'cool': 0.73, 'bright': 0.77}, 
             'output': {'saturation': 1.02, 'temperature': 97, 'brightness': 1.17, 'highlights': 1.12}},
            
            # Correction data (10 pairs)
            {'input': {'gold': 0.94, 'warm': 0.87, 'bright': 0.65}, 
             'output': {'saturation': 1.26, 'temperature': 110, 'brightness': 1.18, 'highlights': 1.12}},
            {'input': {'gold': 0.80, 'warm': 0.91, 'bright': 0.88}, 
             'output': {'saturation': 1.16, 'temperature': 102, 'brightness': 1.06, 'highlights': 1.04}},
            {'input': {'pink': 0.92, 'warm': 0.70, 'bright': 0.70}, 
             'output': {'saturation': 1.20, 'temperature': 106, 'brightness': 1.14, 'highlights': 1.18}},
            {'input': {'pink': 0.80, 'warm': 0.82, 'bright': 0.90}, 
             'output': {'saturation': 1.10, 'temperature': 100, 'brightness': 1.05, 'highlights': 1.08}},
            {'input': {'silver': 0.95, 'cool': 0.90, 'bright': 0.95}, 
             'output': {'saturation': 0.90, 'temperature': 94, 'brightness': 1.20, 'highlights': 1.25}},
            {'input': {'silver': 0.80, 'cool': 0.75, 'bright': 0.80}, 
             'output': {'saturation': 1.00, 'temperature': 99, 'brightness': 1.10, 'highlights': 1.15}},
            {'input': {'silver': 0.90, 'cool': 0.60, 'bright': 0.70}, 
             'output': {'saturation': 1.10, 'temperature': 101, 'brightness': 1.25, 'highlights': 1.20}},
            {'input': {'silver': 0.78, 'cool': 0.78, 'bright': 0.82}, 
             'output': {'saturation': 0.98, 'temperature': 98, 'brightness': 1.12, 'highlights': 1.10}},
            {'input': {'gold': 0.89, 'warm': 0.83, 'bright': 0.90}, 
             'output': {'saturation': 1.17, 'temperature': 103, 'brightness': 1.05, 'highlights': 1.03}},
            {'input': {'pink': 0.84, 'warm': 0.79, 'bright': 0.88}, 
             'output': {'saturation': 1.11, 'temperature': 101, 'brightness': 1.06, 'highlights': 1.07}}
        ]
        
        # Initialize Replicate client
        self.client = self._init_replicate_client()
    
    def _init_replicate_client(self):
        """Initialize Replicate client"""
        try:
            print(f"[{VERSION}] Initializing Replicate client...")
            api_token = os.environ.get('REPLICATE_API_TOKEN')
            if not api_token:
                print(f"[{VERSION}] Warning: REPLICATE_API_TOKEN not found")
                return None
            
            client = replicate.Client(api_token=api_token)
            print(f"[{VERSION}] Replicate client initialized successfully")
            return client
        except Exception as e:
            print(f"[{VERSION}] Failed to initialize Replicate client: {e}")
            return None
    
    def _detect_ring_multi_stage(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Multi-stage ring detection"""
        print(f"[{VERSION}] Starting multi-stage ring detection")
        
        # Stage 1: Try Grounding DINO
        bbox = self._detect_with_grounding_dino(image)
        if bbox:
            print(f"[{VERSION}] ✓ Grounding DINO detection successful")
            return bbox
        
        # Stage 2: Try OWL-ViT
        bbox = self._detect_with_owl_vit(image)
        if bbox:
            print(f"[{VERSION}] ✓ OWL-ViT detection successful")
            return bbox
        
        # Stage 3: OpenCV fallback
        print(f"[{VERSION}] Using OpenCV fallback detection")
        return self._detect_ring_opencv(image)
    
    def _detect_with_grounding_dino(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect ring using Grounding DINO"""
        if not self.client:
            return None
            
        try:
            print(f"[{VERSION}] Stage 1: Grounding DINO detection")
            
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize for API
            max_size = 1024
            if max(img_pil.size) > max_size:
                ratio = max_size / max(img_pil.size)
                new_size = tuple(int(dim * ratio) for dim in img_pil.size)
                img_pil = img_pil.resize(new_size, Image.LANCZOS)
            
            # Convert to base64 WITHOUT padding
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Run detection
            output = self.client.run(
                "lucasjin/grounding-dino:7a93d74ec3a4f62c4530fb971ad9c9fa1a90ac4a4c44f8ee06b2e740ce0e1983",
                input={
                    "image": f"data:image/png;base64,{base64_image}",
                    "prompt": "wedding ring",
                    "box_threshold": 0.3,
                    "text_threshold": 0.25
                }
            )
            
            if output and 'predictions' in output and output['predictions']:
                pred = output['predictions'][0]
                if 'box' in pred:
                    box = pred['box']
                    # Scale back to original size
                    scale_x = image.shape[1] / img_pil.size[0]
                    scale_y = image.shape[0] / img_pil.size[1]
                    
                    x1 = int(box['x_min'] * scale_x)
                    y1 = int(box['y_min'] * scale_y)
                    x2 = int(box['x_max'] * scale_x)
                    y2 = int(box['y_max'] * scale_y)
                    
                    return (x1, y1, x2, y2)
                    
        except Exception as e:
            print(f"[{VERSION}] Grounding DINO error: {e}")
            
        return None
    
    def _detect_with_owl_vit(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect ring using OWL-ViT"""
        if not self.client:
            return None
            
        try:
            print(f"[{VERSION}] Stage 2: OWL-ViT detection")
            
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize for API
            max_size = 768
            if max(img_pil.size) > max_size:
                ratio = max_size / max(img_pil.size)
                new_size = tuple(int(dim * ratio) for dim in img_pil.size)
                img_pil = img_pil.resize(new_size, Image.LANCZOS)
            
            # Convert to base64 WITHOUT padding
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Run detection
            output = self.client.run(
                "adirik/owlvit-base-patch32:5c3c26faa21ee78c98e5c09a11f92e1528c2b6da2f1bb1ab988e1b9f0b1b5a09",
                input={
                    "image": f"data:image/png;base64,{base64_image}",
                    "query": ["wedding ring", "ring", "jewelry"],
                    "score_threshold": 0.1
                }
            )
            
            if output and len(output) > 0:
                best_detection = max(output, key=lambda x: x.get('score', 0))
                if 'box' in best_detection:
                    box = best_detection['box']
                    # Scale back
                    scale_x = image.shape[1] / img_pil.size[0]
                    scale_y = image.shape[0] / img_pil.size[1]
                    
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    return (x1, y1, x2, y2)
                    
        except Exception as e:
            print(f"[{VERSION}] OWL-ViT error: {e}")
            
        return None
    
    def _detect_ring_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback OpenCV detection"""
        try:
            print(f"[{VERSION}] Stage 3: OpenCV detection")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Detect circles
            circles = cv2.HoughCircles(
                filtered,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=100,
                param1=50,
                param2=30,
                minRadius=50,
                maxRadius=500
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                largest_circle = max(circles[0], key=lambda c: c[2])
                x, y, r = largest_circle
                
                # Expand bounding box
                padding = int(r * 0.3)
                x1 = max(0, x - r - padding)
                y1 = max(0, y - r - padding)
                x2 = min(image.shape[1], x + r + padding)
                y2 = min(image.shape[0], y + r + padding)
                
                return (x1, y1, x2, y2)
                
        except Exception as e:
            print(f"[{VERSION}] OpenCV detection error: {e}")
            
        return None
    
    def _remove_background_replicate(self, image: np.ndarray) -> np.ndarray:
        """Remove background using Replicate API"""
        if not self.client:
            print(f"[{VERSION}] Replicate client not available, using original")
            return image
            
        try:
            print(f"[{VERSION}] Removing background with Replicate API")
            
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Convert to base64 WITHOUT padding
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Call Replicate API
            output = self.client.run(
                "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
                input={
                    "image": f"data:image/png;base64,{base64_image}"
                }
            )
            
            if output:
                # Download result
                import requests
                response = requests.get(output)
                if response.status_code == 200:
                    # Convert back to numpy
                    img_no_bg = Image.open(io.BytesIO(response.content))
                    img_no_bg = np.array(img_no_bg)
                    
                    # Ensure BGR format
                    if len(img_no_bg.shape) == 3:
                        if img_no_bg.shape[2] == 4:  # RGBA
                            img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2BGR)
                        elif img_no_bg.shape[2] == 3:  # RGB
                            img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2BGR)
                    
                    print(f"[{VERSION}] Background removed successfully")
                    return img_no_bg
                    
        except Exception as e:
            print(f"[{VERSION}] Background removal error: {e}")
            
        return image
    
    def _analyze_ring_color(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Analyze ring color using ML"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color metrics
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            
            # Convert to RGB for analysis
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            r_mean = np.mean(rgb[:, :, 0])
            g_mean = np.mean(rgb[:, :, 1])
            b_mean = np.mean(rgb[:, :, 2])
            
            # Calculate features
            gold_score = (r_mean > g_mean > b_mean) and (s_mean > 50)
            warm_score = (r_mean + g_mean) / (2 * (b_mean + 1))
            pink_score = (r_mean > g_mean) and (r_mean > b_mean) and (abs(r_mean - g_mean) < 30)
            silver_score = (abs(r_mean - g_mean) < 20) and (abs(g_mean - b_mean) < 20)
            cool_score = b_mean / (r_mean + 1)
            bright_score = v_mean / 255
            
            # Normalize scores - Convert to Python float
            features = {}
            features['gold'] = float(min(1.0, gold_score * warm_score / 2))
            features['warm'] = float(min(1.0, warm_score / 2))
            features['pink'] = float(min(1.0, pink_score * 1.2 if pink_score else 0))
            features['silver'] = float(min(1.0, silver_score * 1.1 if silver_score else 0))
            features['cool'] = float(min(1.0, cool_score))
            features['bright'] = float(bright_score)
            
            # Determine color
            if features['gold'] > 0.7 and features['warm'] > 0.8:
                color = 'yellow_gold'
            elif features['pink'] > 0.6 and features['warm'] > 0.6:
                color = 'rose_gold'
            elif features['silver'] > 0.7 and features['cool'] > 0.6:
                if features['bright'] > 0.8:
                    color = 'white_gold'
                else:
                    color = 'non_plated_white'
            else:
                # Default based on highest scores
                if features['gold'] > max(features['pink'], features['silver']):
                    color = 'yellow_gold'
                elif features['pink'] > features['silver']:
                    color = 'rose_gold'
                else:
                    color = 'white_gold' if features['bright'] > 0.75 else 'non_plated_white'
            
            print(f"[{VERSION}] Color analysis - Detected: {color}")
            print(f"[{VERSION}] Features: gold={features['gold']:.2f}, pink={features['pink']:.2f}, silver={features['silver']:.2f}")
            
            return color, features
            
        except Exception as e:
            print(f"[{VERSION}] Color analysis error: {e}")
            return 'yellow_gold', {'gold': 0.8, 'warm': 0.8, 'bright': 0.8}
    
    def _predict_adjustments(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict adjustments using training data"""
        # Find closest training sample
        min_distance = float('inf')
        best_output = None
        
        for sample in self.training_data:
            distance = 0
            for key in features:
                if key in sample['input']:
                    distance += (features[key] - sample['input'][key]) ** 2
            
            if distance < min_distance:
                min_distance = distance
                best_output = sample['output']
        
        if best_output:
            # Convert all values to Python float
            return {k: float(v) for k, v in best_output.items()}
        else:
            # Default adjustments
            return {
                'saturation': 1.15,
                'temperature': 105.0,
                'brightness': 1.10,
                'highlights': 1.10
            }
    
    def _apply_enhancements(self, image: np.ndarray, adjustments: Dict[str, float]) -> np.ndarray:
        """Apply predicted enhancements"""
        try:
            # Convert to PIL for processing
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply brightness
            if adjustments['brightness'] != 1.0:
                enhancer = ImageEnhance.Brightness(img_pil)
                img_pil = enhancer.enhance(adjustments['brightness'])
            
            # Apply saturation
            if adjustments['saturation'] != 1.0:
                enhancer = ImageEnhance.Color(img_pil)
                img_pil = enhancer.enhance(adjustments['saturation'])
            
            # Convert back to numpy
            enhanced = np.array(img_pil)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # Apply temperature adjustment
            if adjustments['temperature'] != 100:
                temp_factor = adjustments['temperature'] / 100
                if temp_factor > 1:  # Warmer
                    enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * temp_factor, 0, 255)
                else:  # Cooler
                    enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (2 - temp_factor), 0, 255)
            
            # Apply highlights
            if adjustments['highlights'] != 1.0:
                # Create highlights mask
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
                
                # Apply highlight enhancement
                for i in range(3):
                    enhanced[:, :, i] = enhanced[:, :, i] * (1 - mask) + \
                                      np.clip(enhanced[:, :, i] * adjustments['highlights'], 0, 255) * mask
            
            return enhanced.astype(np.uint8)
            
        except Exception as e:
            print(f"[{VERSION}] Enhancement error: {e}")
            return image
    
    def _create_thumbnail(self, image: np.ndarray, size=(1000, 1300)) -> np.ndarray:
        """Create thumbnail with exact size"""
        try:
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Create thumbnail
            img_pil.thumbnail(size, Image.LANCZOS)
            
            # Create white background
            thumb = Image.new('RGB', size, (255, 255, 255))
            
            # Paste centered
            x = (size[0] - img_pil.width) // 2
            y = (size[1] - img_pil.height) // 2
            thumb.paste(img_pil, (x, y))
            
            # Convert back
            return cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"[{VERSION}] Thumbnail error: {e}")
            return image
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Main processing pipeline"""
        print(f"\n[{VERSION}] {'='*50}")
        print(f"[{VERSION}] Starting image processing pipeline")
        print(f"[{VERSION}] Features: Multi-stage detection, 38 training pairs, 1000x1300 thumbnail")
        print(f"[{VERSION}] {'='*50}")
        
        original_shape = image.shape
        print(f"[{VERSION}] Input image shape: {original_shape}")
        
        # Step 1: Multi-stage ring detection
        bbox = self._detect_ring_multi_stage(image)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            ring_roi = image[y1:y2, x1:x2].copy()
            print(f"[{VERSION}] Ring detected at: ({x1},{y1},{x2},{y2})")
        else:
            print(f"[{VERSION}] No ring detected, processing full image")
            ring_roi = image.copy()
            bbox = (0, 0, image.shape[1], image.shape[0])
        
        # Step 2: Remove background
        ring_no_bg = self._remove_background_replicate(ring_roi)
        
        # Step 3: Analyze color
        color, features = self._analyze_ring_color(ring_no_bg)
        
        # Step 4: Predict adjustments
        adjustments = self._predict_adjustments(features)
        print(f"[{VERSION}] Adjustments: {adjustments}")
        
        # Step 5: Apply enhancements
        enhanced_ring = self._apply_enhancements(ring_no_bg, adjustments)
        
        # Step 6: Merge back
        result = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Resize enhanced ring if needed
        if enhanced_ring.shape[:2] != (y2-y1, x2-x1):
            enhanced_ring = cv2.resize(enhanced_ring, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
        
        result[y1:y2, x1:x2] = enhanced_ring
        
        # Step 7: Create thumbnail
        thumbnail = self._create_thumbnail(result, size=(1000, 1300))
        
        # Prepare metadata - all values as Python native types
        metadata = {
            'version': VERSION,
            'color_detected': color,
            'features': features,  # Already converted to float in _analyze_ring_color
            'adjustments': adjustments,  # Already converted to float in _predict_adjustments
            'bbox': [int(x) for x in bbox],  # Convert to int
            'original_size': [int(original_shape[0]), int(original_shape[1])],  # Convert to int
            'training_samples_used': len(self.training_data)
        }
        
        print(f"[{VERSION}] Processing complete")
        print(f"[{VERSION}] {'='*50}\n")
        
        return result, thumbnail, metadata

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Recursively find base64 image data in nested structures"""
    print(f"[{VERSION}] Searching at depth {depth}, type: {type(data)}")
    
    if depth > max_depth:
        return None
    
    if isinstance(data, str):
        # Check if it's base64 data
        if len(data) > 100 and ('/' in data or '+' in data or '=' in data):
            return data
        return None
    
    if isinstance(data, dict):
        # Check common keys first
        for key in ['image_base64', 'image', 'base64', 'data', 'input']:
            if key in data:
                print(f"[{VERSION}] Found image in key: {key}")
                result = find_base64_in_dict(data[key], depth + 1, max_depth)
                if result:
                    return result
        
        # Check all keys
        for key, value in data.items():
            result = find_base64_in_dict(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_base64_in_dict(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array"""
    try:
        print(f"[{VERSION}] Decoding base64 image, length: {len(base64_str)}")
        
        # Remove header if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Clean base64 string
        base64_str = base64_str.strip()
        
        # Add padding if needed (for decoding, not for API calls)
        padding = 4 - (len(base64_str) % 4)
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        try:
            img_bytes = base64.b64decode(base64_str)
            print(f"[{VERSION}] Direct decode successful")
        except:
            # Try without padding
            img_bytes = base64.b64decode(base64_str.rstrip('='))
            print(f"[{VERSION}] Decode successful without padding")
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(img_bytes))
        print(f"[{VERSION}] Image opened successfully: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        print(f"[{VERSION}] Converted to numpy array: {img_array.shape}")
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(f"[{VERSION}] Converted to BGR for OpenCV")
        
        return img_array
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding image: {e}")
        traceback.print_exc()
        raise

def encode_image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """Encode numpy array to base64 string WITHOUT padding"""
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        img_pil = Image.fromarray(image_rgb)
        
        # Save to buffer
        buffer = io.BytesIO()
        img_pil.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Encode to base64 WITHOUT padding
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job["input"]
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image data
        base64_image = find_base64_in_dict(job_input)
        
        if not base64_image:
            return {"output": {"error": "No base64 image data found", "version": VERSION}}
        
        print(f"[{VERSION}] Found image data, length: {len(base64_image)}")
        
        # Decode image
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.shape}")
        
        # Process image
        enhancer = WeddingRingEnhancerV148()
        enhanced, thumbnail, metadata = enhancer.process_image(image)
        
        # Encode results WITHOUT padding
        enhanced_base64 = encode_image_to_base64(enhanced)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        print(f"[{VERSION}] Enhanced image encoded, length: {len(enhanced_base64)}")
        print(f"[{VERSION}] Thumbnail encoded, length: {len(thumbnail_base64)}")
        
        # Return with proper structure
        return {
            "output": {
                "enhanced_image": enhanced_base64,
                "thumbnail": thumbnail_base64,
                "metadata": metadata,
                "success": True,
                "version": VERSION
            }
        }
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(f"[{VERSION}] {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False,
                "version": VERSION
            }
        }

# Print handler info when module loads
print_handler_info()

# RunPod endpoint
runpod.serverless.start({"handler": handler})
