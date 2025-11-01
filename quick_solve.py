# quick_solve.py
"""
Quick problem solver with video generation
No OCR required - just type or select your problem
"""
from ai_vector_solver import AIVectorSolver
from video_solution_generator import VideoSolutionGenerator
from pathlib import Path


# Pre-defined problems from your images
SAMPLE_PROBLEMS = {
    '1': "Matrix equality: [[3, x+y], [x-y, 5]] = [[3, -7], [2, 5]]. Find x and y.",
    '2': "Solve D = 2AB - C where A = [[2, 5], [9, 1]], B = [[4, 1], [3, 7]], C = [[0, 8], [2, 6]]",
    '3': "Find the magnitude of vector <3, 4>",
    '4': "Calculate the cross product of vectors <1, 2, 3> and <4, 5, 6>",
    '5': "Find the dot product of <3, 4, 5> and <1, 2, 3>",
}


def solve_and_create_video(problem_text, video_name="solution.mp4"):
    """Solve problem and create video"""
    print("\n" + "="*70)
    print("QUICK PROBLEM SOLVER WITH VIDEO")
    print("="*70)
    print(f"\nProblem: {problem_text}\n")
    
    # Step 1: Solve
    print("STEP 1: Solving with AI + Vector Analysis...")
    solver = AIVectorSolver()
    solution = solver.solve_problem(problem_text, visualize=True)
    
    # Step 2: Create video
    print(f"\nSTEP 2: Creating video tutorial...")
    viz_dir = Path("data/visualizations")
    viz_files = []
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("solution_*.png"))
        if not viz_files:
            viz_files = list(viz_dir.glob("*.png"))
    
    video_gen = VideoSolutionGenerator(video_name, fps=1)
    video_path = video_gen.create_video_from_solution(solution, viz_files)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE!")
    print("="*70)
    print(f"📹 Video: {video_path}")
    print(f"📊 Report: data/solutions/latest_solution.json")
    print(f"📈 Plots: data/visualizations/")
    
    return video_path


def main():
    import sys
    
    # Check if problem provided via command line
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
        solve_and_create_video(problem)
        return
    
    # Interactive menu
    print("\n" + "="*70)
    print("QUICK PROBLEM SOLVER - INTERACTIVE MODE")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        print("Choose an option:")
        print("\n📚 Sample Problems:")
        for key, problem in SAMPLE_PROBLEMS.items():
            print(f"  {key}. {problem[:60]}...")
        
        print("\n📝 Or:")
        print("  0. Enter custom problem")
        print("  q. Quit")
        print("-"*70)
        
        choice = input("\nYour choice: ").strip()
        
        if choice.lower() in ('q', 'quit', 'exit'):
            print("Goodbye!")
            break
        
        if choice == '0':
            problem = input("\nEnter your problem: ").strip()
            if not problem:
                continue
        elif choice in SAMPLE_PROBLEMS:
            problem = SAMPLE_PROBLEMS[choice]
        else:
            print("Invalid choice. Try again.")
            continue
        
        # Solve and create video
        video_name = f"solution_{choice}.mp4"
        solve_and_create_video(problem, video_name)
        
        another = input("\nSolve another problem? (y/n): ").strip().lower()
        if another != 'y':
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
    