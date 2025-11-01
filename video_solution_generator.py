# video_solution_generator.py
"""
Create animated video solutions from vector analysis and AI explanations
Integrates with ai_vector_solver to generate complete video tutorials
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class VideoSolutionGenerator:
    """Generate educational video from complete solution"""
    
    def __init__(self, output_path="solution_video.mp4", fps=2):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = (1920, 1080)  # HD resolution
        self.bg_color = (255, 255, 255)
        self.text_color = (30, 30, 30)
        self.highlight_color = (0, 100, 200)
        self.accent_color = (220, 80, 80)
        
    def create_video_from_solution(self, solution, visualization_images=None):
        """Create complete video tutorial from solution data"""
        print(f"\n{'='*70}")
        print("CREATING VIDEO SOLUTION")
        print("="*70)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        
        frames = []
        
        # Title frame
        print("✓ Creating title frame...")
        frames.append(self._create_title_frame(solution.get('problem', 'Math Problem')))
        
        # Problem type frame
        print("✓ Creating problem analysis frame...")
        frames.append(self._create_analysis_frame(solution))
        
        # Vector computation frames
        if 'vector_analysis' in solution and solution['vector_analysis'].get('steps'):
            print(f"✓ Creating {len(solution['vector_analysis']['steps'])} computation frames...")
            for step in solution['vector_analysis']['steps']:
                frames.append(self._create_step_frame(step, "Vector Computation"))
        
        # AI explanation frames
        if 'ai_explanation' in solution:
            print("✓ Creating AI explanation frames...")
            explanation_frames = self._create_explanation_frames(solution['ai_explanation'])
            frames.extend(explanation_frames)
        
        # Visualization frames
        if visualization_images:
            print(f"✓ Adding {len(visualization_images)} visualization frames...")
            for viz_img in visualization_images:
                frames.append(self._create_visualization_frame(viz_img))
        
        # Final answer frame
        print("✓ Creating final answer frame...")
        final_answer = solution.get('final_answer', 'See explanation above')
        frames.append(self._create_final_frame(final_answer))
        
        # Write all frames to video
        print(f"✓ Writing {len(frames)} frames to video...")
        for frame in frames:
            # Hold each frame for duration based on fps
            for _ in range(int(self.fps * 3)):  # 3 seconds per frame
                video.write(frame)
        
        video.release()
        print(f"✓ Video created: {self.output_path}")
        return self.output_path
    
    def _create_title_frame(self, problem_text):
        """Create animated title frame"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            subtitle_font = ImageFont.truetype("arial.ttf", 48)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Title
        title = "Math Problem Solution"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.frame_size[0] - title_width) // 2
        draw.text((title_x, 200), title, fill=self.highlight_color, font=title_font)
        
        # Problem text (wrapped)
        wrapped = textwrap.fill(problem_text, width=60)
        problem_bbox = draw.textbbox((0, 0), wrapped, font=subtitle_font)
        problem_width = problem_bbox[2] - problem_bbox[0]
        problem_x = max(100, (self.frame_size[0] - problem_width) // 2)
        
        # Draw box around problem
        padding = 40
        draw.rectangle([
            problem_x - padding, 
            400 - padding,
            problem_x + problem_width + padding,
            400 + (problem_bbox[3] - problem_bbox[1]) + padding
        ], outline=self.highlight_color, width=5)
        
        draw.text((problem_x, 400), wrapped, fill=self.text_color, font=subtitle_font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _create_analysis_frame(self, solution):
        """Create frame showing problem analysis"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            header_font = ImageFont.truetype("arial.ttf", 60)
            text_font = ImageFont.truetype("arial.ttf", 42)
        except:
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Header
        draw.text((100, 100), "Problem Analysis", fill=self.highlight_color, font=header_font)
        
        y = 250
        
        # Problem type
        ptype = solution.get('problem_type', 'unknown')
        draw.text((150, y), f"Type: {ptype.upper()}", fill=self.text_color, font=text_font)
        y += 100
        
        # Components
        if 'vector_analysis' in solution:
            va = solution['vector_analysis']
            if 'steps' in va and va['steps']:
                draw.text((150, y), f"Steps: {len(va['steps'])}", fill=self.text_color, font=text_font)
                y += 100
        
        # Draw checkmark
        draw.text((150, y + 100), "✓ Ready to solve", fill=(0, 150, 0), font=text_font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _create_step_frame(self, step, title="Solution Step"):
        """Create frame for a solution step"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 56)
            text_font = ImageFont.truetype("arial.ttf", 40)
            calc_font = ImageFont.truetype("arial.ttf", 48)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            calc_font = ImageFont.load_default()
        
        # Step header
        step_num = step.get('step_number', '?')
        desc = step.get('description', 'Step')
        header = f"Step {step_num}: {desc}"
        draw.text((100, 100), header, fill=self.highlight_color, font=title_font)
        
        y = 250
        
        # Calculation (in box)
        if 'calculation' in step:
            calc = str(step['calculation'])
            wrapped_calc = textwrap.fill(calc, width=50)
            
            # Draw calculation box
            calc_bbox = draw.textbbox((100, y), wrapped_calc, font=calc_font)
            padding = 30
            draw.rectangle([
                100 - padding,
                y - padding,
                calc_bbox[2] + padding,
                calc_bbox[3] + padding
            ], outline=self.accent_color, width=4, fill=(250, 250, 255))
            
            draw.text((100, y), wrapped_calc, fill=self.text_color, font=calc_font)
            y = calc_bbox[3] + 100
        
        # Explanation
        if 'explanation' in step:
            expl = step['explanation']
            wrapped_expl = textwrap.fill(expl, width=70)
            draw.text((100, y), wrapped_expl, fill=(80, 80, 80), font=text_font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _create_explanation_frames(self, explanation_text):
        """Create frames from AI explanation (split into chunks)"""
        frames = []
        
        # Split explanation into paragraphs
        paragraphs = [p.strip() for p in explanation_text.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs):
            img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            
            try:
                header_font = ImageFont.truetype("arial.ttf", 56)
                text_font = ImageFont.truetype("arial.ttf", 38)
            except:
                header_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Header
            draw.text((100, 80), "AI Tutor Explanation", fill=self.highlight_color, font=header_font)
            draw.text((100, 160), f"Part {i + 1} of {len(paragraphs)}", fill=(120, 120, 120), font=text_font)
            
            # Content
            wrapped = textwrap.fill(para, width=75)
            draw.text((100, 280), wrapped, fill=self.text_color, font=text_font)
            
            frames.append(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
        
        return frames
    
    def _create_visualization_frame(self, viz_image_path):
        """Create frame with visualization image"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        
        # Load visualization
        viz_img = cv2.imread(str(viz_image_path))
        if viz_img is None:
            return img
        
        # Resize to fit frame
        h, w = viz_img.shape[:2]
        max_h, max_w = 800, 1600
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        viz_resized = cv2.resize(viz_img, (new_w, new_h))
        
        # Center in frame
        y_offset = (self.frame_size[1] - new_h) // 2
        x_offset = (self.frame_size[0] - new_w) // 2
        
        img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = viz_resized
        
        # Add title
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        draw.text((100, 50), "Visual Representation", fill=self.highlight_color, font=font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _create_final_frame(self, final_answer):
        """Create final answer frame"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            answer_font = ImageFont.truetype("arial.ttf", 60)
        except:
            title_font = ImageFont.load_default()
            answer_font = ImageFont.load_default()
        
        # Title
        title = "Final Answer"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.frame_size[0] - title_width) // 2
        draw.text((title_x, 250), title, fill=self.highlight_color, font=title_font)
        
        # Answer
        wrapped_answer = textwrap.fill(str(final_answer), width=50)
        answer_bbox = draw.textbbox((0, 0), wrapped_answer, font=answer_font)
        answer_width = answer_bbox[2] - answer_bbox[0]
        answer_x = max(100, (self.frame_size[0] - answer_width) // 2)
        
        # Draw box around answer
        padding = 60
        draw.rectangle([
            answer_x - padding,
            500 - padding,
            answer_x + answer_width + padding,
            500 + (answer_bbox[3] - answer_bbox[1]) + padding
        ], fill=(240, 255, 240), outline=(0, 150, 0), width=6)
        
        draw.text((answer_x, 500), wrapped_answer, fill=(0, 100, 0), font=answer_font)
        
        # Checkmark
        draw.text((self.frame_size[0] // 2 - 50, 750), "✓", fill=(0, 150, 0), font=title_font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    """Test video generation"""
    import sys
    from ai_vector_solver import AIVectorSolver
    
    if len(sys.argv) < 2:
        print("Usage: python video_solution_generator.py <problem_text>")
        print("\nExample:")
        print('  python video_solution_generator.py "Find magnitude of vector <3, 4>"')
        return 1
    
    problem = " ".join(sys.argv[1:])
    
    # Solve problem
    print("Solving problem...")
    solver = AIVectorSolver()
    solution = solver.solve_problem(problem, visualize=True)
    
    # Get visualization files
    viz_dir = Path("data/visualizations")
    viz_files = list(viz_dir.glob("solution_*.png")) if viz_dir.exists() else []
    
    # Generate video
    video_gen = VideoSolutionGenerator("math_solution.mp4", fps=1)
    video_path = video_gen.create_video_from_solution(solution, viz_files)
    
    print(f"\n{'='*70}")
    print(f"✓ Video created: {video_path}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())