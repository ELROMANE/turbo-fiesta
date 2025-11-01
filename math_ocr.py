# math_ocr.py
import os
import shutil
import pytesseract
from dotenv import load_dotenv
import cv2
import numpy as np
from pathlib import Path

class MathOCR:
    def __init__(self):
        # Load environment variables from a local .env if present
        try:
            load_dotenv()
        except Exception:
            # Non-fatal: dotenv is optional
            pass

        # Configure multiple Tesseract configs for math recognition
        self.tess_configs = [
            # Standard equation config with expanded math symbols
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-=(){}[]xyz<>∑∫∞πΣαβγδθλμ∏∪∩∈∀∃≤≥≠≈→⇒',
            # Single-line equation mode
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-=(){}[]xyz<>∑∫∞πΣαβγδθλμ∏∪∩∈∀∃≤≥≠≈→⇒',
            # Single word/symbol mode for isolated characters
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-=(){}[]xyz<>∑∫∞πΣαβγδθλμ∏∪∩∈∀∃≤≥≠≈→⇒',
            # Treat image as single character (for large isolated symbols)
            r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789+-=(){}[]xyz<>∑∫∞πΣαβγδθλμ∏∪∩∈∀∃≤≥≠≈→⇒',
            # Specialized math mode for subscripts and superscripts
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789+-=(){}[]xyz<>∑∫∞πΣαβγδθλμ∏∪∩∈∀∃≤≥≠≈→⇒'
        ]

        # Try to locate tesseract executable from environment, PATH or common install locations
        tess_from_env = os.environ.get("TESSERACT_CMD")
        tess_in_path = shutil.which("tesseract")
        common_locations = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        tess_cmd = tess_from_env or tess_in_path
        if not tess_cmd:
            for p in common_locations:
                if Path(p).exists():
                    tess_cmd = p
                    break

        if tess_cmd:
            # Configure pytesseract to use the discovered executable
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
            self.tesseract_available = True
            self.tesseract_path = tess_cmd
        else:
            self.tesseract_available = False
            self.tesseract_path = None

    def preprocess_image(self, image_path, debug_output=False):
        """Enhance image for better OCR with math-specific processing"""
        # Load and validate image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques optimized for math
        preprocessed = []

        # 1. Basic binary threshold with light denoising (good for clear prints)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(("binary", binary))

        # 2. Adaptive threshold with morphology (good for uneven lighting)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        preprocessed.append(("adaptive", morph))

        # 3. Edge-preserving smoothing with sharpening (good for handwritten)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(smooth, -1, kernel_sharp)
        _, enhanced = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(("enhanced", enhanced))

        # 4. High contrast with symbol isolation (good for mixed content)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (3,3), 0)
        _, high_contrast = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(("contrast", high_contrast))
        
        # 5. Scale-space analysis for subscripts and superscripts
        height = gray.shape[0]
        scales = [1.0, 1.5, 2.0]  # Multiple scales to capture different sizes
        for scale in scales:
            target_height = int(height * scale)
            scaled = cv2.resize(gray, (int(gray.shape[1] * scale), target_height))
            denoised_scale = cv2.fastNlMeansDenoising(scaled, None, 10, 7, 21)
            _, binary_scale = cv2.threshold(denoised_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed.append((f"scale_{scale}", binary_scale))

        if debug_output:
            # Save debug images to see preprocessing results
            debug_dir = Path(image_path).parent / "debug_ocr"
            debug_dir.mkdir(exist_ok=True)
            base_name = Path(image_path).stem
            for name, img in preprocessed:
                out_path = debug_dir / f"{base_name}_{name}.png"
                cv2.imwrite(str(out_path), img)

        # Return all variants for OCR attempts
        return [img for _, img in preprocessed]
    
    def extract_math_problem(self, image_path):
        """Extract math problem text from image using multiple preprocessing and OCR passes"""
        if not getattr(self, "tesseract_available", False):
            print("OCR Error: tesseract is not installed or it's not in your PATH. See README file for more information.")
            return None

        # Get multiple preprocessed versions of the image
        processed_images = self.preprocess_image(image_path, debug_output=False)
        
        best_text = None
        best_confidence = 0
        
        try:
            # Try each preprocessing variant with each OCR config
            for img in processed_images:
                for config in self.tess_configs:
                    # Get OCR text and confidence data
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence for non-empty text
                    confs = [conf for conf, text in zip(data['conf'], data['text']) if str(text).strip()]
                    if confs:
                        avg_conf = sum(confs) / len(confs)
                        text = ' '.join(data['text'])
                        cleaned = self.clean_math_text(text)
                        
                        if self.validate_math_problem(cleaned) and avg_conf > best_confidence:
                            best_text = cleaned
                            best_confidence = avg_conf
            
            return best_text

        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def clean_math_text(self, text):
        """Clean OCR output for math problems with enhanced math symbol handling"""
        # Basic cleanup
        text = ' '.join(text.split())
        
        # Fix common OCR mistakes and normalize math symbols
        replacements = {
            '—': '-',  # Em dash to minus
            '–': '-',  # En dash to minus
            '−': '-',  # Unicode minus to ASCII minus
            '×': '*',  # Times symbol to asterisk
            '÷': '/',  # Division symbol to forward slash
            '·': '*',  # Centered dot to asterisk
            '∗': '*',  # Alternative asterisk
            '=': '=',  # Normalize equals
            '{': '(',  # Normalize brackets
            '}': ')',
            '[': '(',
            ']': ')',
            '⎜': '|',  # Vertical bars
            '⎟': '|',
            '∈': 'in',  # Set notation
            '≤': '<=',  # Inequalities
            '≥': '>=',
            '≠': '!=',
            '≈': '~=',
            '∑': 'sum',  # Special functions
            '∫': 'int',
            '→': '->',  # Arrows
            '⇒': '=>',
            'Σ': 'sum',
            'π': 'pi',
            '∞': 'inf',
            '\n': ' ',  # Replace newlines with spaces
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        # Remove unnecessary spaces around operators
        text = text.replace(' + ', '+').replace(' - ', '-').replace(' * ', '*').replace(' / ', '/')
        text = text.replace(' = ', '=').replace('( ', '(').replace(' )', ')')
        
        # Clean up any remaining whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def validate_math_problem(self, text):
        """Check if text looks like a valid math problem"""
        if not text or len(text) < 2:
            return False
            
        # Extended math indicators including variables and functions
        math_indicators = [
            '+', '-', '*', '/', '=', '<', '>', '!',  # Operators
            'x', 'y', 'z', 'n', 'i', 'j', 'k',      # Common variables
            'sin', 'cos', 'tan', 'log', 'ln',        # Functions
            'solve', 'find', 'calculate',            # Problem keywords
            'sum', 'int', 'dx', 'dy',                # Calculus terms
            '(', ')', '[', ']',                      # Grouping
            'lim', 'inf', 'sup', 'det',             # Advanced math terms
            'alpha', 'beta', 'gamma', 'theta',       # Greek letters
            'integral', 'series', 'sequence'         # Math concepts
        ]
        
        # Text must contain at least one math indicator
        if not any(indicator in text.lower() for indicator in math_indicators):
            return False
            
        # Text should have a reasonable ratio of math symbols to total length
        math_chars = sum(1 for c in text if c in '+-*/=()<>[]{}0123456789xy')
        if math_chars / len(text) < 0.15:  # At least 15% math characters
            return False
            
        return True
