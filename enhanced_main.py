# Fix for enhanced_main.py - patch the create_solution_video method
# Add this to your EnhancedMathSolver class in enhanced_main.py

def create_solution_video(self, solution, output_path):
    """Create animated video of the solution - FIXED VERSION"""
    print(f"\n{'='*60}")
    print("CREATING SOLUTION VIDEO")
    print(f"{'='*60}")
    
    # Fix: Ensure solution has required 'problem' key
    if 'problem' not in solution:
        # Try to get problem from steps or create a default
        if 'steps' in solution and solution['steps']:
            problem_text = solution['steps'][0].get('calculation', 'Math Problem')
        else:
            problem_text = "Math Problem Solution"
        solution['problem'] = problem_text
    
    # Also ensure other required keys exist
    if 'final_answer' not in solution:
        solution['final_answer'] = 'See steps above'
    
    if 'steps' not in solution:
        solution['steps'] = []
    
    animator = MathVideoAnimator(output_path)
    
    try:
        video_path = animator.create_video_from_solution(solution)
        print(f"✓ Video created: {video_path}")
        return video_path
    except Exception as e:
        print(f"Warning: Video creation failed: {e}")
        print("Skipping video generation...")
        return None