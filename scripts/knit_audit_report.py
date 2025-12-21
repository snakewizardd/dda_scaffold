#!/usr/bin/env python3
"""
Knit all audit images into a single composite report image.
"""

import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image

AUDIT_DIR = Path("data/audit")
OUTPUT_PATH = AUDIT_DIR / "audit_visual_report.png"

# Images to include (in order)
IMAGE_FILES = [
    "trust_matrix_heatmap.png",
    "trust_deltas_timeline.png",
    "trust_cause_by_phase.png",
    "trust_flow_network.png",
    "trust_network_centrality.png",
]

def knit_images():
    """Combine images into a single vertical report."""
    images = []
    
    for fname in IMAGE_FILES:
        fpath = AUDIT_DIR / fname
        if fpath.exists():
            img = Image.open(fpath)
            images.append((fname, img))
            print(f"✓ Loaded: {fname} ({img.size[0]}x{img.size[1]})")
        else:
            print(f"⚠ Missing: {fname}")
    
    if not images:
        print("No images found!")
        return
    
    # Calculate total dimensions
    max_width = max(img.size[0] for _, img in images)
    total_height = sum(img.size[1] for _, img in images)
    
    # Add padding and title space
    padding = 20
    title_height = 60
    total_height += (len(images) + 1) * padding + title_height
    
    # Create composite
    composite = Image.new('RGB', (max_width + 2*padding, total_height), color='white')
    
    # Add title (using simple text positioning)
    y_offset = title_height
    
    # Paste images
    for fname, img in images:
        # Center horizontally
        x_offset = (max_width - img.size[0]) // 2 + padding
        composite.paste(img, (x_offset, y_offset))
        y_offset += img.size[1] + padding
    
    # Save
    composite.save(OUTPUT_PATH, quality=95)
    print(f"\n✓ Saved composite: {OUTPUT_PATH}")
    print(f"  Dimensions: {composite.size[0]}x{composite.size[1]}")


if __name__ == "__main__":
    knit_images()
