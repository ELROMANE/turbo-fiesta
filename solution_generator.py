# solution_generator.py
import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import json
from pathlib import Path

class MathSolutionGenerator:
    def __init__(self):
        self.api_key = self.get_api_key()
        self.client = Cerebras(api_key=self.api_key) if self.api_key else None
    
    def get_api_key(self):
        """Get API key from environment"""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
        return os.environ.get("CEREBRAS_API_KEY")
    
    def generate_step_by_step_solution(self, math_problem):
        """Generate detailed step-by-step math solution"""
        prompt = f"""
        You are an expert math tutor. Solve this problem step by step as if teaching a student.
        Provide clear explanations for each step and show the reasoning.
        
        Problem: {math_problem}
        
        Format your response as JSON with this structure:
        {{
            "problem": "original problem",
            "steps": [
                {{
                    "step_number": 1,
                    "description": "what we're doing in this step",
                    "calculation": "mathematical work",
                    "explanation": "why we're doing this"
                }}
            ],
            "final_answer": "the solution"
        }}
        """
        
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct",
            )
            
            solution_text = resp.choices[0].message.content
            return self.parse_solution(solution_text)
            
        except Exception as e:
            print(f"API Error: {e}")
            return self.create_fallback_solution(math_problem)
    
    def parse_solution(self, solution_text):
        """Parse AI response into structured solution"""
        try:
            # Try to extract JSON from response
            json_start = solution_text.find('{')
            json_end = solution_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = solution_text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback: create basic structure
        return self.create_fallback_solution(solution_text)
    
    def create_fallback_solution(self, problem):
        """Create solution structure when parsing fails"""
        return {
            "problem": problem,
            "steps": [
                {
                    "step_number": 1,
                    "description": "Analyze the problem",
                    "calculation": problem,
                    "explanation": "Understanding what needs to be solved"
                }
            ],
            "final_answer": "Solution requires manual calculation"
        }