# video_animator.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

class MathVideoAnimator:
    def __init__(self, output_path="math_solution.mp4"):
        self.output_path = output_path
        self.frame_size = (1280, 720)
        self.background_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        self.highlight_color = (0, 100, 200)
        
    def create_video_from_solution(self, solution_data):
        """Create video from structured solution"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.output_path, fourcc, 1.0, self.frame_size)
        
        # Title frame
        title_frame = self.create_title_frame(solution_data["problem"])
        video.write(title_frame)
        
        # Step frames
        for step in solution_data["steps"]:
            step_frame = self.create_step_frame(step)
            video.write(step_frame)
            
            # Add drawing animation for calculations
            if "calculation" in step and step["calculation"]:
                calc_frame = self.animate_calculation(step["calculation"])
                video.write(calc_frame)
        
        # Final answer frame
        final_frame = self.create_final_frame(solution_data["final_answer"])
        video.write(final_frame)
        
        video.release()
        return self.output_path
    
    def create_title_frame(self, problem):
        """Create title frame with the math problem"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 32)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw title
        title = "Math Problem Solution"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_x = (self.frame_size[0] - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 100), title, fill=self.text_color, font=font_large)
        
        # Draw problem
        wrapped_problem = textwrap.fill(problem, width=50)
        problem_bbox = draw.textbbox((0, 0), wrapped_problem, font=font_small)
        problem_x = (self.frame_size[0] - (problem_bbox[2] - problem_bbox[0])) // 2
        draw.text((problem_x, 250), wrapped_problem, fill=self.highlight_color, font=font_small)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def create_step_frame(self, step):
        """Create frame for each solution step"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_normal = ImageFont.truetype("arial.ttf", 28)
        except:
            font_large = ImageFont.load_default()
            font_normal = ImageFont.load_default()
        
        # Step header
        step_title = f"Step {step['step_number']}: {step['description']}"
        draw.text((50, 50), step_title, fill=self.highlight_color, font=font_large)
        
        # Calculation
        if "calculation" in step:
            calc_text = f"Calculation: {step['calculation']}"
            wrapped_calc = textwrap.fill(calc_text, width=60)
            draw.text((50, 150), wrapped_calc, fill=self.text_color, font=font_normal)
        
        # Explanation
        if "explanation" in step:
            wrapped_explanation = textwrap.fill(step['explanation'], width=70)
            draw.text((50, 300), wrapped_explanation, fill=(100, 100, 100), font=font_normal)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def animate_calculation(self, calculation):
        """Animate mathematical calculations with sketching effect"""
        # This would be enhanced with proper drawing animations
        # For now, create a visual representation
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw calculation in center with box
        calc_bbox = draw.textbbox((0, 0), calculation, font=font)
        calc_x = (self.frame_size[0] - (calc_bbox[2] - calc_bbox[0])) // 2
        calc_y = (self.frame_size[1] - (calc_bbox[3] - calc_bbox[1])) // 2
        
        # Draw box around calculation
        padding = 20
        draw.rectangle([
            calc_x - padding, calc_y - padding,
            calc_x + (calc_bbox[2] - calc_bbox[0]) + padding,
            calc_y + (calc_bbox[3] - calc_bbox[1]) + padding
        ], outline=self.highlight_color, width=3)
        
        draw.text((calc_x, calc_y), calculation, fill=self.text_color, font=font)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def create_final_frame(self, final_answer):
        """Create final frame with the answer"""
        img = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_answer = ImageFont.truetype("arial.ttf", 64)
        except:
            font_large = ImageFont.load_default()
            font_answer = ImageFont.load_default()
        
        # Final answer title
        title = "Final Answer"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_x = (self.frame_size[0] - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 200), title, fill=self.text_color, font=font_large)
        
        # The answer
        answer_bbox = draw.textbbox((0, 0), final_answer, font=font_answer)
        answer_x = (self.frame_size[0] - (answer_bbox[2] - answer_bbox[0])) // 2
        draw.text((answer_x, 350), final_answer, fill=self.highlight_color, font=font_answer)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)