import os

pages_dir = r"d:\Documents\Medical Image Processing\pages"

renames = [
    ("1_🔒_Anonymization.py", "1_Anonymization.py"),
    ("2_🧠_Segmentation.py", "2_Segmentation.py"),
    ("3_🔬_CT_Recon.py", "3_CT_Reconstruction.py"),
    ("4_🧲_MRI_Recon.py", "4_MRI_Reconstruction.py"),
    ("5_⚙️_Preprocessing.py", "5_Preprocessing.py"),
]

for old_name, new_name in renames:
    old_path = os.path.join(pages_dir, old_name)
    new_path = os.path.join(pages_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✓ Renamed: {old_name} -> {new_name}")
    else:
        print(f"✗ Not found: {old_name}")

print("\nDone!")
