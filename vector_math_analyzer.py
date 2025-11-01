"""
Enhanced Math Problem Analyzer with Vector Analysis and Graphing
Integrates with your existing Cerebras-based solution generator
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from pathlib import Path
import json


class VectorMathAnalyzer:
    """Analyzes math problems and identifies vector/matrix operations"""
    
    def __init__(self):
        self.problem_type = None
        self.components = {}
        
    def analyze_problem(self, problem_text):
        """Identify problem type and extract mathematical components"""
        analysis = {
            'type': 'unknown',
            'components': {},
            'operations': [],
            'visualization_needed': False
        }
        
        # Detect problem types
        if self._is_vector_problem(problem_text):
            analysis['type'] = 'vector'
            analysis['visualization_needed'] = True
            analysis['components'] = self._extract_vectors(problem_text)
            
        elif self._is_matrix_problem(problem_text):
            analysis['type'] = 'matrix'
            analysis['components'] = self._extract_matrices(problem_text)
            
        elif self._contains_calculus(problem_text):
            analysis['type'] = 'calculus'
            analysis['visualization_needed'] = True
            
        elif self._is_sequence_series(problem_text):
            analysis['type'] = 'sequence/series'
            analysis['visualization_needed'] = True
            
        # Identify operations
        analysis['operations'] = self._identify_operations(problem_text)
        
        return analysis
    
    def _is_vector_problem(self, text):
        """Check if problem involves vectors"""
        vector_indicators = [
            r'vector', r'<[^>]+>', r'\([^)]+\)', 
            r'i\s*\+\s*j', r'cross product', r'dot product',
            r'magnitude', r'direction', r'unit vector'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in vector_indicators)
    
    def _is_matrix_problem(self, text):
        """Check if problem involves matrices"""
        matrix_indicators = [
            r'matrix', r'matrices', r'\[.*\]', 
            r'determinant', r'det\(', r'transpose',
            r'inverse', r'eigenvalue'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in matrix_indicators)
    
    def _contains_calculus(self, text):
        """Check for calculus operations"""
        calculus_indicators = [
            r'd[xy]/d[xy]', r'integral', r'∫', r'derivative',
            r'lim', r'limit', r"f'", r'∂'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in calculus_indicators)
    
    def _is_sequence_series(self, text):
        """Check for sequences or series"""
        indicators = [r'a_n', r'sequence', r'series', r'∑', r'sum']
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in indicators)
    
    def _extract_vectors(self, text):
        """Extract vector components from text"""
        vectors = {}
        
        # Pattern for <a, b, c> notation
        angle_bracket_pattern = r'<([^>]+)>'
        matches = re.findall(angle_bracket_pattern, text)
        for i, match in enumerate(matches):
            components = [float(x.strip()) for x in match.split(',')]
            vectors[f'v{i+1}'] = np.array(components)
        
        # Pattern for (a, b, c) notation
        paren_pattern = r'\(([^)]+)\)'
        matches = re.findall(paren_pattern, text)
        for match in matches:
            try:
                components = [float(x.strip()) for x in match.split(',')]
                if len(components) in [2, 3]:
                    key = f'v{len(vectors)+1}'
                    vectors[key] = np.array(components)
            except ValueError:
                continue
        
        return vectors
    
    def _extract_matrices(self, text):
        """Extract matrix components from text"""
        matrices = {}
        
        # Simple pattern for matrix notation [a b; c d]
        matrix_pattern = r'\[([^\]]+)\]'
        matches = re.findall(matrix_pattern, text)
        
        for i, match in enumerate(matches):
            try:
                rows = match.split(';')
                matrix_data = []
                for row in rows:
                    elements = [float(x.strip()) for x in row.split()]
                    matrix_data.append(elements)
                matrices[f'M{i+1}'] = np.array(matrix_data)
            except ValueError:
                continue
        
        return matrices
    
    def _identify_operations(self, text):
        """Identify mathematical operations needed"""
        operations = []
        
        operation_map = {
            'addition': [r'\+', r'add', r'sum'],
            'subtraction': [r'-', r'subtract', r'difference'],
            'multiplication': [r'\*', r'multiply', r'product'],
            'cross_product': [r'cross', r'×'],
            'dot_product': [r'dot', r'·'],
            'magnitude': [r'magnitude', r'length', r'\|\|'],
            'normalize': [r'unit', r'normalize'],
            'determinant': [r'det', r'determinant'],
            'transpose': [r'transpose', r'\^T'],
        }
        
        for op_name, patterns in operation_map.items():
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                operations.append(op_name)
        
        return operations


class VectorGrapher:
    """Creates visualizations for vector and matrix problems"""
    
    def __init__(self, output_dir='data/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_vectors_2d(self, vectors, title="Vector Visualization", filename=None):
        """Plot 2D vectors"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (name, vec) in enumerate(vectors.items()):
            if len(vec) == 2:
                ax.quiver(0, 0, vec[0], vec[1], 
                         angles='xy', scale_units='xy', scale=1,
                         color=colors[i % len(colors)], 
                         width=0.006, label=name)
                # Add label at vector tip
                ax.text(vec[0]*1.1, vec[1]*1.1, name, fontsize=12)
        
        # Set axis limits
        all_coords = np.array([v for v in vectors.values() if len(v) == 2])
        if len(all_coords) > 0:
            max_val = np.abs(all_coords).max() * 1.5
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved 2D plot to {filepath}")
        
        return fig
    
    def plot_vectors_3d(self, vectors, title="3D Vector Visualization", filename=None):
        """Plot 3D vectors"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (name, vec) in enumerate(vectors.items()):
            if len(vec) == 3:
                ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                         color=colors[i % len(colors)],
                         arrow_length_ratio=0.1, label=name, linewidth=2)
                # Add label at vector tip
                ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, name, fontsize=12)
        
        # Set axis limits
        all_coords = np.array([v for v in vectors.values() if len(v) == 3])
        if len(all_coords) > 0:
            max_val = np.abs(all_coords).max() * 1.5
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_zlim(-max_val, max_val)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {filepath}")
        
        return fig
    
    def plot_vector_operations(self, vec1, vec2, operation='add', filename=None):
        """Visualize vector operations (addition, subtraction)"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Original vectors
        ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', 
                 scale=1, color='red', width=0.006, label='v1')
        ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', 
                 scale=1, color='blue', width=0.006, label='v2')
        
        # Result based on operation
        if operation == 'add':
            result = vec1 + vec2
            ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy',
                     scale=1, color='green', width=0.008, label='v1 + v2')
            # Show parallelogram
            ax.quiver(vec1[0], vec1[1], vec2[0], vec2[1], angles='xy', 
                     scale_units='xy', scale=1, color='blue', 
                     width=0.003, alpha=0.5, linestyle='dashed')
            ax.quiver(vec2[0], vec2[1], vec1[0], vec1[1], angles='xy',
                     scale_units='xy', scale=1, color='red',
                     width=0.003, alpha=0.5, linestyle='dashed')
            
        elif operation == 'subtract':
            result = vec1 - vec2
            ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy',
                     scale=1, color='green', width=0.008, label='v1 - v2')
        
        # Set axis properties
        all_vecs = np.array([vec1, vec2, result])
        max_val = np.abs(all_vecs).max() * 1.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Vector {operation.capitalize()}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved operation plot to {filepath}")
        
        return fig
    
    def visualize_matrix(self, matrix, title="Matrix Heatmap", filename=None):
        """Visualize matrix as heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(matrix, cmap='RdBu', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved matrix heatmap to {filepath}")
        
        return fig


class VectorSolutionGenerator:
    """Generate step-by-step solutions for vector problems"""
    
    def solve_vector_problem(self, analysis):
        """Generate solution steps based on analysis"""
        solution = {
            'problem_type': analysis['type'],
            'steps': [],
            'visualizations': []
        }
        
        if analysis['type'] == 'vector':
            vectors = analysis['components']
            operations = analysis['operations']
            
            # Add vector identification step
            solution['steps'].append({
                'step_number': 1,
                'description': 'Identify vectors',
                'calculation': str(vectors),
                'explanation': f'Problem contains {len(vectors)} vector(s)'
            })
            
            # Calculate operations
            if 'magnitude' in operations and vectors:
                for name, vec in vectors.items():
                    mag = np.linalg.norm(vec)
                    solution['steps'].append({
                        'step_number': len(solution['steps']) + 1,
                        'description': f'Calculate magnitude of {name}',
                        'calculation': f'||{name}|| = sqrt({" + ".join([f"{x}^2" for x in vec])}) = {mag:.2f}',
                        'explanation': 'Magnitude is the length of the vector'
                    })
            
            if 'dot_product' in operations and len(vectors) >= 2:
                vec_list = list(vectors.values())
                dot_prod = np.dot(vec_list[0], vec_list[1])
                solution['steps'].append({
                    'step_number': len(solution['steps']) + 1,
                    'description': 'Calculate dot product',
                    'calculation': f'v1 · v2 = {dot_prod:.2f}',
                    'explanation': 'Dot product measures vector alignment'
                })
            
            if 'cross_product' in operations and len(vectors) >= 2:
                vec_list = list(vectors.values())
                if len(vec_list[0]) == 3 and len(vec_list[1]) == 3:
                    cross = np.cross(vec_list[0], vec_list[1])
                    solution['steps'].append({
                        'step_number': len(solution['steps']) + 1,
                        'description': 'Calculate cross product',
                        'calculation': f'v1 × v2 = {cross}',
                        'explanation': 'Cross product is perpendicular to both vectors'
                    })
        
        return solution


# Example usage function
def analyze_and_solve_vector_problem(problem_text, image_path=None):
    """Main function to analyze and solve vector problems with visualization"""
    
    print("="*60)
    print("VECTOR MATH PROBLEM ANALYZER")
    print("="*60)
    
    # Step 1: Analyze the problem
    analyzer = VectorMathAnalyzer()
    analysis = analyzer.analyze_problem(problem_text)
    
    print(f"\nProblem Type: {analysis['type']}")
    print(f"Operations Needed: {', '.join(analysis['operations'])}")
    print(f"Visualization Needed: {analysis['visualization_needed']}")
    
    # Step 2: Generate solution
    solver = VectorSolutionGenerator()
    solution = solver.solve_vector_problem(analysis)
    
    print("\nSOLUTION STEPS:")
    for step in solution['steps']:
        print(f"\nStep {step['step_number']}: {step['description']}")
        print(f"  Calculation: {step['calculation']}")
        print(f"  Explanation: {step['explanation']}")
    
    # Step 3: Create visualizations
    if analysis['visualization_needed'] and analysis['components']:
        grapher = VectorGrapher()
        
        if analysis['type'] == 'vector':
            vectors = analysis['components']
            
            # Check dimensionality
            dims = set(len(v) for v in vectors.values())
            
            if 2 in dims:
                vecs_2d = {k: v for k, v in vectors.items() if len(v) == 2}
                grapher.plot_vectors_2d(vecs_2d, 
                                       title="Vector Problem Visualization",
                                       filename="vector_problem_2d.png")
            
            if 3 in dims:
                vecs_3d = {k: v for k, v in vectors.items() if len(v) == 3}
                grapher.plot_vectors_3d(vecs_3d,
                                       title="3D Vector Problem Visualization",
                                       filename="vector_problem_3d.png")
    
    return analysis, solution


# Test examples
if __name__ == "__main__":
    # Example 1: 2D Vector Addition
    problem1 = "Given vectors v1 = <3, 4> and v2 = <-1, 2>, find their sum and magnitude"
    analyze_and_solve_vector_problem(problem1)
    
    # Example 2: 3D Cross Product
    problem2 = "Calculate the cross product of vectors <1, 2, 3> and <4, 5, 6>"
    analyze_and_solve_vector_problem(problem2)
    
    # Example 3: Matrix from your image
    problem3 = "Matrix equality: [[3, x+y], [x-y, 5]] = [[3, -7], [2, 5]]. Find x and y."
    analyze_and_solve_vector_problem(problem3)