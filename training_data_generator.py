import os
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


class MathTrainingDataGenerator:
    def __init__(self, base_dir='data/training'):
        self.base_dir = base_dir
        self.categories = ['sequences', 'series', 'calculus', 'matrices']
        # Use width,height that matches validator expectations (width between 100-400, height between 400-1200)
        self.image_size = (200, 600)  # Width, Height

    def generate_background(self):
        """Generate a random background with slight noise and color variation."""
        background = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
        noise = np.random.normal(0, 2, background.shape).astype(np.uint8)
        background = cv2.add(background, noise)
        return background

    def get_math_expressions(self, category):
        expressions = {
            # sequences and subscripts/superscripts
            'sequences': [
                'a_n = n^2',
                'a_{n+1} = a_n + 1',
                'a_n = 2^n',
                'a_n = n!',
                'a_n = 1/n',
                'a_n = (-1)^n',
                'b_k = k^3 + 3k^2',
                'c_{n+2} = c_n + c_{n+1}',
            ],
            # series including summation notation, partial sums, convergence
            'series': [
                '∑_{n=1}^∞ 1/n^2',
                '∑_{n=0}^∞ 1/2^n',
                '∑_{n=1}^N n^2',
                '∑_{n=1}^∞ (-1)^{n}/n',
                'S_N = \sum_{n=1}^N a_n',
                '∑_{k=0}^n C(n,k)',
            ],
            # calculus: derivatives, integrals, limits, differential equations
            'calculus': [
                'dy/dx = 3x^2',
                'd^2y/dx^2 + y = 0',
                '∫_0^1 x^2 dx',
                'lim_{x→∞} (1 + 1/x)^x',
                'f\'(x) = \\frac{d}{dx} sin x',
                '∮_C F·dr = 0',
                '∂^2 u/∂x^2 + ∂^2 u/∂y^2 = 0',
            ],
            # matrices: brackets, determinants, transpose
            'matrices': [
                '[a b; c d]',
                '[1 0; 0 1]',
                'det(A) = ad - bc',
                'A^T',
                'tr(A)',
                '\n'.join(['[1 2 3]', '[4 5 6]', '[7 8 9]']),
            ],
        }
        return expressions.get(category, [])

    def generate_sample_set(self, samples_per_category=50, out_dir=None):
        """Generate a smaller sample set and write a manifest (CSV + JSON)."""
        import csv
        if out_dir is None:
            out_dir = os.path.join(self.base_dir, 'sample')
        os.makedirs(out_dir, exist_ok=True)

        manifest = []
        for category in self.categories:
            category_dir = os.path.join(out_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            expressions = self.get_math_expressions(category)
            for i in range(samples_per_category):
                expression = random.choice(expressions)
                img = self.create_image(expression)
                filename = f"{category}_{i+1}.png"
                filepath = os.path.join(category_dir, filename)
                img.save(filepath)
                truth_path = os.path.join(category_dir, f"{category}_{i+1}.txt")
                with open(truth_path, 'w', encoding='utf-8') as f:
                    f.write(expression)
                manifest.append({'image': os.path.relpath(filepath), 'label': expression, 'category': category})

        # write CSV
        csv_path = os.path.join(out_dir, 'manifest.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['image', 'label', 'category'])
            writer.writeheader()
            for row in manifest:
                writer.writerow(row)

        # write JSON
        import json
        json_path = os.path.join(out_dir, 'manifest.json')
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(manifest, jf, ensure_ascii=False, indent=2)

        print(f"Sample set generated at {out_dir} with manifest.csv and manifest.json")

    def create_image(self, expression):
        # Create background (note: PIL expects HxW for numpy->Image.fromarray)
        bg = self.generate_background()
        img = Image.fromarray(bg)

        # Prefer a bundled math font in ./fonts/math_font.ttf for reproducible rendering.
        # Fallback order: bundled font -> system Arial -> PIL default
        font = None
        try:
            bundled = os.path.join(os.path.dirname(__file__), 'fonts', 'math_font.ttf')
            if os.path.exists(bundled):
                font = ImageFont.truetype(bundled, int(self.image_size[1] * 0.08))
        except Exception:
            font = None

        if font is None:
            try:
                sys_font = r"C:\Windows\Fonts\arial.ttf"
                if os.path.exists(sys_font):
                    font = ImageFont.truetype(sys_font, int(self.image_size[1] * 0.08))
            except Exception:
                font = None

        if font is None:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)

        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), expression, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = max(0, (self.image_size[0] - text_width) // 2)
        y = max(0, (self.image_size[1] - text_height) // 2)

        # Draw the text with a small stroke by drawing multiple offsets to ensure visibility
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((x + dx, y + dy), expression, fill=(0, 0, 0), font=font)

        return img

    def generate_dataset(self, samples_per_category=100):
        for category in self.categories:
            print(f"Generating {samples_per_category} samples for {category}...")
            category_dir = os.path.join(self.base_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            expressions = self.get_math_expressions(category)
            if not expressions:
                print(f"No expressions defined for category {category}, skipping.")
                continue

            for i in range(samples_per_category):
                expression = random.choice(expressions)
                img = self.create_image(expression)

                filename = f"{category}_{i+1}.png"
                filepath = os.path.join(category_dir, filename)
                img.save(filepath)

                truth_file = os.path.join(category_dir, f"{category}_{i+1}.txt")
                with open(truth_file, 'w', encoding='utf-8') as f:
                    f.write(expression)

                # Tiny sleep to make sure filesystem has flushed
                time.sleep(0.01)

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{samples_per_category} for {category}")


if __name__ == '__main__':
    gen = MathTrainingDataGenerator()
    gen.generate_dataset()
