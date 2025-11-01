# test_vector_analyzer.py
"""
Simple test script for vector analysis without camera/OCR dependencies
Use this to test vector analysis and visualization features
"""
from vector_math_analyzer import VectorMathAnalyzer, VectorGrapher, VectorSolutionGenerator
import json
from pathlib import Path


def test_vector_problem(problem_text):
    """Test vector analysis and visualization"""
    print("\n" + "="*60)
    print("VECTOR MATH ANALYZER TEST")
    print("="*60)
    print(f"\nProblem: {problem_text}\n")
    
    # Step 1: Analyze
    print("="*60)
    print("ANALYZING PROBLEM")
    print("="*60)
    analyzer = VectorMathAnalyzer()
    analysis = analyzer.analyze_problem(problem_text)
    
    print(f"Problem Type: {analysis['type']}")
    print(f"Operations: {', '.join(analysis['operations']) if analysis['operations'] else 'None detected'}")
    print(f"Visualization Needed: {analysis['visualization_needed']}")
    
    if analysis['components']:
        print("\nComponents Found:")
        for name, value in analysis['components'].items():
            print(f"  {name}: {value}")
    
    # Step 2: Solve
    print("\n" + "="*60)
    print("GENERATING SOLUTION")
    print("="*60)
    solver = VectorSolutionGenerator()
    solution = solver.solve_vector_problem(analysis)
    
    print("\nSolution Steps:")
    for step in solution.get('steps', []):
        print(f"\n  Step {step['step_number']}: {step['description']}")
        print(f"    Calculation: {step.get('calculation', 'N/A')}")
        print(f"    Explanation: {step.get('explanation', 'N/A')}")
    
    # Step 3: Visualize
    if analysis['visualization_needed'] and analysis['components']:
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        grapher = VectorGrapher()
        viz_files = []
        
        if analysis['type'] == 'vector':
            vectors = analysis['components']
            dims = set(len(v) for v in vectors.values())
            
            if 2 in dims:
                vecs_2d = {k: v for k, v in vectors.items() if len(v) == 2}
                filename = f"test_2d_{len(viz_files)}.png"
                grapher.plot_vectors_2d(vecs_2d, title=problem_text[:50], filename=filename)
                viz_files.append(filename)
                print(f"✓ Created 2D plot: {filename}")
            
            if 3 in dims:
                vecs_3d = {k: v for k, v in vectors.items() if len(v) == 3}
                filename = f"test_3d_{len(viz_files)}.png"
                grapher.plot_vectors_3d(vecs_3d, title=problem_text[:50], filename=filename)
                viz_files.append(filename)
                print(f"✓ Created 3D plot: {filename}")
        
        if analysis['type'] == 'matrix':
            for name, matrix in analysis['components'].items():
                filename = f"test_matrix_{name}.png"
                grapher.visualize_matrix(matrix, title=f"Matrix {name}", filename=filename)
                viz_files.append(filename)
                print(f"✓ Created matrix heatmap: {filename}")
        
        print(f"\nVisualization files saved in: {grapher.output_dir}")
    
    # Save report
    report = {
        'problem': problem_text,
        'analysis': analysis,
        'solution': solution
    }
    
    report_dir = Path("data/test_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "test_report.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        report_json = {
            'problem': report['problem'],
            'analysis': {
                'type': report['analysis']['type'],
                'operations': report['analysis']['operations'],
                'visualization_needed': report['analysis']['visualization_needed'],
                'components': {k: v.tolist() if hasattr(v, 'tolist') else str(v) 
                              for k, v in report['analysis']['components'].items()}
            },
            'solution': report['solution']
        }
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved: {report_path}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # If user provides command line argument, use that
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
        test_vector_problem(problem)
    else:
        # Run all test cases
        print("\n\n*** TEST 1: Simple 2D Vector Magnitude ***")
        test_vector_problem("Find the magnitude of vector <3, 4>")
        
        print("\n\n*** TEST 2: 3D Vector Cross Product ***")
        test_vector_problem("Calculate the cross product of vectors <1, 2, 3> and <4, 5, 6>")
        
        print("\n\n*** TEST 3: Vector Addition ***")
        test_vector_problem("Add vectors <2, 3> and <5, -1>")
        
        print("\n\n*** TEST 4: Matrix Problem ***")
        test_vector_problem("Matrix [[3, 7], [2, 5]] find determinant")
        
        print("\n\n*** TEST 5: Dot Product ***")
        test_vector_problem("Find the dot product of <1, 2, 3> and <4, 5, 6>")
        