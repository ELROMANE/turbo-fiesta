# solve_from_image.py
"""
Extract math problem from image and solve it with AI + vector analysis
"""
import sys
from pathlib import Path
from math_ocr import MathOCR
from ai_vector_solver import AIVectorSolver


def solve_from_image(image_path):
    """Extract problem from image and solve it"""
    print("\n" + "="*70)
    print("MATH PROBLEM SOLVER FROM IMAGE")
    print("="*70)
    print(f"\nImage: {image_path}\n")
    
    # Step 1: Extract text from image
    print("="*70)
    print("STEP 1: EXTRACTING TEXT FROM IMAGE")
    print("="*70)
    
    ocr = MathOCR()
    problem_text = ocr.extract_math_problem(image_path)
    
    if not problem_text:
        print("❌ Could not extract text from image.")
        print("\nTips:")
        print("  - Make sure the image is clear and well-lit")
        print("  - Ensure text is horizontal and not skewed")
        print("  - Try preprocessing the image (crop, enhance contrast)")
        return None
    
    print(f"✓ Extracted text: {problem_text}")
    
    # Step 2: Solve the problem
    print(f"\n{'='*70}")
    print("STEP 2: SOLVING THE PROBLEM")
    print("="*70)
    
    solver = AIVectorSolver()
    solution = solver.solve_problem(problem_text)
    
    return solution


def main():
    if len(sys.argv) < 2:
        print("Usage: python solve_from_image.py <image_path>")
        print("\nExample:")
        print("  python solve_from_image.py matrix.jpg")
        print("  python solve_from_image.py path/to/problem.png")
        return 1
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    solution = solve_from_image(image_path)
    
    if solution:
        print("\n" + "="*70)
        print("✓ COMPLETE!")
        print("="*70)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())