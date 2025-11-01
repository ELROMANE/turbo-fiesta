# ai_vector_solver.py
"""
Complete math solver combining vector analysis with Cerebras AI
Works with text input, no camera/OCR dependencies required
"""
import os
import json
from pathlib import Path
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from vector_math_analyzer import VectorMathAnalyzer, VectorGrapher, VectorSolutionGenerator


class AIVectorSolver:
    """Combines vector analysis with AI-powered explanations"""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.client = Cerebras(api_key=self.api_key) if self.api_key else None
        self.vector_analyzer = VectorMathAnalyzer()
        self.vector_solver = VectorSolutionGenerator()
        self.grapher = VectorGrapher()
        
    def get_api_key(self):
        """Get API key from environment"""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
        return os.environ.get("CEREBRAS_API_KEY")
    
    def solve_with_ai(self, problem_text, analysis, vector_solution):
        """Use Cerebras AI to generate detailed explanation"""
        if not self.client:
            print("Warning: No Cerebras API key found. Using vector solver only.")
            return vector_solution
        
        # Create enhanced prompt with analysis context
        components_str = ""
        if analysis['components']:
            components_str = "\n".join([f"  {k}: {v}" for k, v in analysis['components'].items()])
        
        prompt = f"""
You are an expert math tutor. Solve this {analysis['type']} problem step by step.

Problem: {problem_text}

Mathematical Components Detected:
{components_str}

Operations Needed: {', '.join(analysis['operations'])}

Provide:
1. A clear explanation of what the problem asks
2. Step-by-step solution with calculations
3. Final answer
4. Key concepts involved

Format your response as clear, numbered steps.
"""
        
        try:
            print("Consulting AI for detailed explanation...")
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
            )
            
            ai_explanation = resp.choices[0].message.content
            
            # Combine vector solution with AI explanation
            combined_solution = {
                'problem': problem_text,
                'problem_type': analysis['type'],
                'vector_analysis': vector_solution,
                'ai_explanation': ai_explanation,
                'final_answer': self._extract_answer(ai_explanation)
            }
            
            return combined_solution
            
        except Exception as e:
            print(f"AI Error: {e}")
            print("Falling back to vector solver only.")
            return vector_solution
    
    def _extract_answer(self, text):
        """Try to extract final answer from AI response"""
        import re
        # Look for common answer patterns
        patterns = [
            r'final answer[:\s]+([^\n]+)',
            r'answer[:\s]+([^\n]+)',
            r'therefore[,\s]+([^\n]+)',
            r'result[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[-1] if sentences else "See explanation above"
    
    def solve_problem(self, problem_text, visualize=True):
        """Complete problem solving pipeline"""
        print("\n" + "="*70)
        print("AI-ENHANCED VECTOR PROBLEM SOLVER")
        print("="*70)
        print(f"\nProblem: {problem_text}\n")
        
        # Step 1: Analyze structure
        print("="*70)
        print("STEP 1: ANALYZING PROBLEM STRUCTURE")
        print("="*70)
        analysis = self.vector_analyzer.analyze_problem(problem_text)
        
        print(f"✓ Problem Type: {analysis['type']}")
        print(f"✓ Operations: {', '.join(analysis['operations']) if analysis['operations'] else 'Standard math'}")
        print(f"✓ Visualization Needed: {analysis['visualization_needed']}")
        
        if analysis['components']:
            print(f"\nMathematical Components:")
            for name, value in analysis['components'].items():
                print(f"  {name}: {value}")
        
        # Step 2: Vector solution
        print(f"\n{'='*70}")
        print("STEP 2: COMPUTING VECTOR/MATRIX OPERATIONS")
        print("="*70)
        vector_solution = self.vector_solver.solve_vector_problem(analysis)
        
        if vector_solution['steps']:
            for step in vector_solution['steps']:
                print(f"\n  • {step['description']}")
                if 'calculation' in step:
                    print(f"    Result: {step['calculation']}")
        
        # Step 3: AI Enhancement
        print(f"\n{'='*70}")
        print("STEP 3: GENERATING AI EXPLANATION")
        print("="*70)
        complete_solution = self.solve_with_ai(problem_text, analysis, vector_solution)
        
        if 'ai_explanation' in complete_solution:
            print("\nAI Tutor Explanation:")
            print("-" * 70)
            print(complete_solution['ai_explanation'])
            print("-" * 70)
        
        # Step 4: Visualization
        if visualize and analysis['visualization_needed'] and analysis['components']:
            print(f"\n{'='*70}")
            print("STEP 4: CREATING VISUALIZATIONS")
            print("="*70)
            
            viz_files = []
            
            if analysis['type'] == 'vector':
                vectors = analysis['components']
                dims = set(len(v) for v in vectors.values())
                
                if 2 in dims:
                    vecs_2d = {k: v for k, v in vectors.items() if len(v) == 2}
                    filename = f"solution_2d.png"
                    self.grapher.plot_vectors_2d(vecs_2d, title=problem_text[:50], filename=filename)
                    viz_files.append(filename)
                    print(f"✓ Created 2D plot: {filename}")
                
                if 3 in dims:
                    vecs_3d = {k: v for k, v in vectors.items() if len(v) == 3}
                    filename = f"solution_3d.png"
                    self.grapher.plot_vectors_3d(vecs_3d, title=problem_text[:50], filename=filename)
                    viz_files.append(filename)
                    print(f"✓ Created 3D plot: {filename}")
            
            if analysis['type'] == 'matrix':
                for name, matrix in analysis['components'].items():
                    filename = f"solution_matrix_{name}.png"
                    self.grapher.visualize_matrix(matrix, title=f"Matrix {name}", filename=filename)
                    viz_files.append(filename)
                    print(f"✓ Created matrix heatmap: {filename}")
            
            if viz_files:
                print(f"\nVisualization files saved in: {self.grapher.output_dir}")
                complete_solution['visualizations'] = viz_files
        
        # Step 5: Save report
        self._save_report(complete_solution)
        
        # Summary
        print(f"\n{'='*70}")
        print("✓ SOLUTION COMPLETE!")
        print("="*70)
        if 'final_answer' in complete_solution:
            print(f"\nFinal Answer: {complete_solution['final_answer']}")
        
        return complete_solution
    
    def _save_report(self, solution):
        """Save complete solution report"""
        report_dir = Path("data/solutions")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "latest_solution.json"
        
        # Convert numpy arrays to lists for JSON
        def serialize(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            return obj
        
        serializable_solution = serialize(solution)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_solution, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Full report saved: {report_path}")


def main():
    import sys
    
    solver = AIVectorSolver()
    
    if len(sys.argv) > 1:
        # Problem from command line
        problem = " ".join(sys.argv[1:])
        solver.solve_problem(problem)
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("AI-ENHANCED VECTOR PROBLEM SOLVER")
        print("Interactive Mode")
        print("="*70)
        print("\nEnter your math problem (or 'quit' to exit):")
        print("Examples:")
        print("  - Find the magnitude of vector <3, 4>")
        print("  - Calculate cross product of <1,2,3> and <4,5,6>")
        print("  - Matrix [[2, 3], [1, 4]] find determinant")
        print("  - Solve: 2x + 3y = 7 and x - y = 1")
        
        while True:
            try:
                problem = input("\nProblem: ").strip()
                
                if problem.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                if not problem:
                    continue
                
                solver.solve_problem(problem)
                
                print("\n" + "-"*70)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()