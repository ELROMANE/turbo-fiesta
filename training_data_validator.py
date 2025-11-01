import cv2
import numpy as np
from pathlib import Path
import re

class TrainingDataValidator:
    def __init__(self, base_dir='data/training'):
        self.base_dir = base_dir
        self.categories = ['sequences', 'series', 'calculus', 'matrices']
        self.min_image_size = (100, 400)  # Minimum width, height
        self.max_image_size = (400, 1200)  # Maximum width, height
        
    def validate_image_quality(self, image_path):
        """Check if image meets quality standards."""
        try:
            # Read image with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Failed to load image"
            
            # Check image size
            height, width = img.shape[:2]
            if (width < self.min_image_size[0] or height < self.min_image_size[1] or
                width > self.max_image_size[0] or height > self.max_image_size[1]):
                return False, f"Image size {width}x{height} outside allowed range"
            
            # Convert to grayscale for further analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Check if image is not empty/blank using non-white pixel ratio
            nonwhite = np.count_nonzero(gray < 250)
            ratio = nonwhite / float(gray.size)
            # Require at least 0.5% non-white pixels (adjustable)
            if ratio < 0.005:
                return False, f"Image appears to be mostly blank or solid (nonwhite_ratio={ratio:.4f})"
            
            # Check contrast
            min_val, max_val = np.percentile(gray, [1, 99])
            if max_val - min_val < 50:
                return False, "Image has insufficient contrast"
            
            # Check for excessive noise
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            noise = np.std(gray.astype(float) - blur.astype(float))
            if noise > 30:
                return False, "Image contains excessive noise"
            
            return True, "Image meets quality standards"
            
        except Exception as e:
            return False, f"Error analyzing image: {str(e)}"
    
    def validate_expression(self, expression, category=None):
        """Validate mathematical expression format."""
        # Common patterns that should be present in valid math expressions
        pattern_map = {
            'sequences': r'a_[n|k]|n[+\-*/^]|\d+',
            'series': r'∑|\+|n[+\-*/^]|\d+',
            'calculus': r'[d|∂]/|∫|lim|dx|dy',
            'matrices': r'[\[\]]|det|tr|\d+|[a-zA-Z]'
        }
        
        # Basic validation checks
        if not expression or len(expression) < 2:
            return False, "Expression too short"
        
        # Check for balanced parentheses and brackets
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for char in expression:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop() != pairs[char]:
                    return False, "Unbalanced parentheses or brackets"
        
        if stack:
            return False, "Unbalanced parentheses or brackets"
        
        # Check for invalid character sequences
        if re.search(r'[+\-*/]{2,}', expression):
            return False, "Invalid operator sequence"
        
        # Category-specific validation
        if category and category in pattern_map:
            if not re.search(pattern_map[category], expression):
                return False, f"Expression does not match {category} pattern"
        
        return True, "Expression format is valid"
    
    def validate_dataset(self):
        """Validate entire training dataset."""
        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'category_stats': {}
        }
        
        for category in self.categories:
            category_path = Path(self.base_dir) / category
            if not category_path.exists():
                print(f"Warning: Category directory {category} not found")
                continue
            
            category_stats = {
                'total': 0,
                'valid': 0,
                'invalid': []
            }
            
            # Validate each image and its corresponding ground truth
            for img_path in category_path.glob('*.png'):
                results['total_files'] += 1
                category_stats['total'] += 1
                
                # Get corresponding truth file
                truth_path = img_path.with_suffix('.txt')
                
                # Validate image quality
                img_valid, img_message = self.validate_image_quality(img_path)
                
                # Validate ground truth if image is valid
                truth_valid = False
                if img_valid and truth_path.exists():
                    with open(truth_path, 'r', encoding='utf-8') as f:
                        expression = f.read().strip()
                        truth_valid, truth_message = self.validate_expression(expression)
                else:
                    truth_message = "Missing ground truth file"
                
                # Record results
                if img_valid and truth_valid:
                    results['valid_files'] += 1
                    category_stats['valid'] += 1
                else:
                    error_info = {
                        'file': str(img_path),
                        'image_error': None if img_valid else img_message,
                        'truth_error': None if truth_valid else truth_message
                    }
                    results['invalid_files'].append(error_info)
                    category_stats['invalid'].append(error_info)
            
            results['category_stats'][category] = category_stats
        
        return results
    
    def print_validation_report(self, results):
        """Print a formatted validation report."""
        print("\n=== Training Data Validation Report ===")
        print(f"\nTotal Files: {results['total_files']}")
        print(f"Valid Files: {results['valid_files']}")
        print(f"Invalid Files: {len(results['invalid_files'])}")
        
        print("\nCategory Statistics:")
        for category, stats in results['category_stats'].items():
            print(f"\n{category.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Valid: {stats['valid']}")
            print(f"  Invalid: {len(stats['invalid'])}")
        
        if results['invalid_files']:
            print("\nDetailed Error Report:")
            for error in results['invalid_files']:
                print(f"\nFile: {error['file']}")
                if error['image_error']:
                    print(f"  Image Error: {error['image_error']}")
                if error['truth_error']:
                    print(f"  Truth Error: {error['truth_error']}")

if __name__ == "__main__":
    validator = TrainingDataValidator()
    results = validator.validate_dataset()
    validator.print_validation_report(results)