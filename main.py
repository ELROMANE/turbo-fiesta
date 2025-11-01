# main.py
import argparse
import os
from camera_capture import MathProblemCapturer
from math_ocr import MathOCR
from solution_generator import MathSolutionGenerator
from video_animator import MathVideoAnimator

def main():
    parser = argparse.ArgumentParser(description="Math Problem to Video Solution Generator")
    parser.add_argument("--camera", action="store_true", help="Capture problem from camera")
    parser.add_argument("--image", type=str, help="Path to math problem image")
    parser.add_argument("--output", type=str, default="math_solution.mp4", help="Output video path")
    
    args = parser.parse_args()
    
    # Step 1: Capture/load math problem
    capturer = MathProblemCapturer()
    if args.camera:
        image_path = capturer.capture_from_webcam()
    elif args.image:
        image_path = args.image
    else:
        print("Please specify --camera or --image")
        return
    
    # Step 2: Extract math problem text
    print("Extracting math problem from image...")
    ocr = MathOCR()
    math_problem = ocr.extract_math_problem(image_path)
    
    if not math_problem:
        print("Could not extract math problem. Please try with a clearer image.")
        return
    
    print(f"Extracted problem: {math_problem}")
    
    # Step 3: Generate AI solution
    print("Generating step-by-step solution...")
    solver = MathSolutionGenerator()
    solution = solver.generate_step_by_step_solution(math_problem)
    
    # Step 4: Create video
    print("Creating solution video...")
    animator = MathVideoAnimator(args.output)
    video_path = animator.create_video_from_solution(solution)
    
    print(f"Video created: {video_path}")
    print(f"Problem: {math_problem}")
    print(f"Solution: {solution.get('final_answer', 'Check video for details')}")

if __name__ == "__main__":
    main()