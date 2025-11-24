"""Quick script to reorganize pages directory"""

import os
import shutil
from pathlib import Path

# Create advanced directory
advanced_dir = Path("pages/advanced")
advanced_dir.mkdir(exist_ok=True)
print(f"✅ Created: {advanced_dir}")

# Files to move
files_to_move = [
    ("pages/1_Anonymization.py", "pages/advanced/Anonymization.py"),
    ("pages/2_Segmentation.py", "pages/advanced/Segmentation.py"),
    ("pages/3_CT_Reconstruction.py", "pages/advanced/CT_Reconstruction.py"),
    ("pages/4_MRI_Reconstruction.py", "pages/advanced/MRI_Reconstruction.py"),
    ("pages/5_Preprocessing.py", "pages/advanced/Preprocessing.py"),
    ("pages/6_Registration.py", "pages/advanced/Registration.py"),
    ("pages/7_Computer_Vision.py", "pages/advanced/Computer_Vision.py"),
]

# Move files
for src, dst in files_to_move:
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"✅ Moved: {src} → {dst}")
    else:
        print(f"⚠️ Not found: {src}")

print("\n🎉 Reorganization complete!")
print(f"📁 New structure:")
print(f"  pages/")
print(f"  └── advanced/")
for _, dst in files_to_move:
    if os.path.exists(dst):
        print(f"      ├── {Path(dst).name}")
