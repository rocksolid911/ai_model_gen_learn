"""
Generate sample datasets for testing the ML pipeline.

This script creates:
1. Sample news articles CSV for text classification
2. Sample images for image classification
"""

import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_news_data(num_samples: int = 1000, output_path: str = "data/news.csv"):
    """
    Generate synthetic news articles for classification.

    Args:
        num_samples: Number of samples to generate
        output_path: Path to save CSV file
    """
    print(f"Generating {num_samples} news articles...")

    # Templates and keywords for each class
    templates = {
        'neutral': [
            "The {subject} announced {action} today at a press conference.",
            "According to recent reports, {subject} has {action}.",
            "Officials stated that {subject} plans to {action}.",
            "{subject} released a statement regarding {action}.",
            "The latest data shows that {subject} continues to {action}.",
        ],
        'left': [
            "Progressive activists celebrated as {subject} finally {action}.",
            "In a bold move for social justice, {subject} decided to {action}.",
            "Critics from the right condemned {subject} for choosing to {action}.",
            "The forward-thinking {subject} embraced change by {action}.",
            "Advocates praised {subject} for taking a stand and {action}.",
        ],
        'right': [
            "Conservative leaders applauded {subject} for {action}.",
            "In defense of traditional values, {subject} chose to {action}.",
            "Patriots celebrated as {subject} defended freedom by {action}.",
            "The sensible {subject} rejected radical proposals and instead {action}.",
            "Law and order advocates supported {subject} decision to {action}.",
        ],
        'propaganda': [
            "BREAKING: The evil {subject} MUST BE STOPPED from {action}!",
            "SHOCKING revelation: {subject} is secretly planning to {action}!!",
            "Wake up people! {subject} is conspiring to {action} and destroy everything!",
            "The corrupt {subject} wants to {action} - Don't let them win!",
            "URGENT: Share this! {subject} caught red-handed trying to {action}!!!",
        ]
    }

    subjects = [
        "government", "administration", "Congress", "Senate", "president",
        "mayor", "governor", "committee", "department", "agency",
        "officials", "lawmakers", "representatives", "council"
    ]

    actions = [
        "implement new policies", "address the situation", "make changes",
        "review regulations", "announce reforms", "propose legislation",
        "conduct investigations", "allocate funding", "develop programs",
        "establish guidelines", "modify procedures", "update standards"
    ]

    records = []

    for i in range(num_samples):
        # Randomly select class
        label = random.choice(['neutral', 'left', 'right', 'propaganda'])

        # Generate title
        template = random.choice(templates[label])
        subject = random.choice(subjects)
        action = random.choice(actions)

        title = template.format(subject=subject, action=action)

        # Generate content (expanded version)
        content_sentences = []

        # Add opening
        content_sentences.append(title)

        # Add 3-5 additional sentences
        for _ in range(random.randint(3, 5)):
            template = random.choice(templates[label])
            subject = random.choice(subjects)
            action = random.choice(actions)
            sentence = template.format(subject=subject, action=action)
            content_sentences.append(sentence)

        content = " ".join(content_sentences)

        records.append({
            'id': i,
            'title': title,
            'content': content,
            'label': label
        })

    # Create DataFrame and save
    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Generated {num_samples} news articles")
    print(f"✓ Saved to {output_path}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())


def generate_image_data(
    num_classes: int = 5,
    samples_per_class: int = 100,
    image_size: tuple = (128, 128),
    output_dir: str = "data/images"
):
    """
    Generate synthetic images for classification.

    Creates colored images with different shapes and patterns for each class.

    Args:
        num_classes: Number of image classes
        samples_per_class: Number of samples per class
        image_size: Size of generated images (width, height)
        output_dir: Directory to save images
    """
    print(f"\nGenerating {num_classes} classes with {samples_per_class} images each...")

    class_names = [
        "circles", "squares", "triangles", "stars", "hexagons",
        "diamonds", "crosses", "hearts", "clouds", "arrows"
    ][:num_classes]

    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (255, 192, 203),  # Pink
        (0, 128, 128),    # Teal
    ]

    os.makedirs(output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"  Generating class '{class_name}'...")

        for sample_idx in range(samples_per_class):
            # Create image with random background
            bg_color = tuple(np.random.randint(200, 256, 3))
            img = Image.new('RGB', image_size, bg_color)
            draw = ImageDraw.Draw(img)

            # Draw shapes based on class
            main_color = colors[class_idx]
            num_shapes = random.randint(3, 7)

            for _ in range(num_shapes):
                # Random position and size
                x = random.randint(10, image_size[0] - 40)
                y = random.randint(10, image_size[1] - 40)
                size = random.randint(15, 35)

                # Add some color variation
                color_variation = tuple(
                    max(0, min(255, c + random.randint(-30, 30)))
                    for c in main_color
                )

                if class_name == "circles":
                    draw.ellipse([x, y, x+size, y+size], fill=color_variation, outline=(0,0,0))

                elif class_name == "squares":
                    draw.rectangle([x, y, x+size, y+size], fill=color_variation, outline=(0,0,0))

                elif class_name == "triangles":
                    points = [(x+size//2, y), (x, y+size), (x+size, y+size)]
                    draw.polygon(points, fill=color_variation, outline=(0,0,0))

                elif class_name == "stars":
                    # Simple star shape
                    center_x, center_y = x + size//2, y + size//2
                    points = []
                    for i in range(10):
                        angle = i * 36 * np.pi / 180
                        r = size//2 if i % 2 == 0 else size//4
                        px = center_x + r * np.cos(angle)
                        py = center_y + r * np.sin(angle)
                        points.append((px, py))
                    draw.polygon(points, fill=color_variation, outline=(0,0,0))

                elif class_name == "hexagons":
                    # Simple hexagon
                    center_x, center_y = x + size//2, y + size//2
                    points = []
                    for i in range(6):
                        angle = i * 60 * np.pi / 180
                        px = center_x + size//2 * np.cos(angle)
                        py = center_y + size//2 * np.sin(angle)
                        points.append((px, py))
                    draw.polygon(points, fill=color_variation, outline=(0,0,0))

                else:
                    # Default to circle for other shapes
                    draw.ellipse([x, y, x+size, y+size], fill=color_variation, outline=(0,0,0))

            # Add some noise for variety
            img_array = np.array(img)
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Save image
            img_path = os.path.join(class_dir, f"{class_name}_{sample_idx:04d}.jpg")
            img.save(img_path, 'JPEG', quality=90)

        print(f"    ✓ Created {samples_per_class} images for '{class_name}'")

    total_images = num_classes * samples_per_class
    print(f"\n✓ Generated {total_images} images across {num_classes} classes")
    print(f"✓ Saved to {output_dir}/")


def main():
    """Generate all sample datasets."""
    print("="*70)
    print("Sample Data Generator")
    print("="*70)
    print("\nThis script generates synthetic data for testing the ML pipeline.")
    print("The data is not real but mimics the structure of actual datasets.\n")

    # Generate news data
    print("\n[1/2] Generating News Dataset")
    print("-" * 70)
    generate_news_data(num_samples=1000, output_path="data/news.csv")

    # Generate image data
    print("\n[2/2] Generating Image Dataset")
    print("-" * 70)
    generate_image_data(
        num_classes=5,
        samples_per_class=100,
        image_size=(128, 128),
        output_dir="data/images"
    )

    print("\n" + "="*70)
    print("Sample Data Generation Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/news.csv (1000 news articles)")
    print("  - data/images/ (500 images across 5 classes)")
    print("\nYou can now train the models:")
    print("  1. python src/training/train_news_baseline.py")
    print("  2. python src/training/train_news_tf.py")
    print("  3. python src/training/train_image_cnn.py")


if __name__ == "__main__":
    main()
