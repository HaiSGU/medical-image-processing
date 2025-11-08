"""Auto-fix common lint errors in pages/"""

import re
from pathlib import Path


def fix_file(file_path):
    """Fix common lint errors."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Fix specific long lines
    fixes = [
        # File 3
        (
            r'method = st\.selectbox\("Phương pháp:", \["FBP", "SART"\], help="Thuật toán tái tạo"\)',
            'method = st.selectbox(\n        "Phương pháp:",\n        ["FBP", "SART"],\n        help="Thuật toán tái tạo"\n    )'
        ),
        (
            r'reconstructed = reconstructor\.reconstruct_fbp\(filter_name=filter_type\)',
            'reconstructed = reconstructor.reconstruct_fbp(\n                    filter_name=filter_type\n                )'
        ),
        (
            r'if st\.button\("🔬 Tái tạo", type="primary", use_container_width=True\):',
            'if st.button(\n                "🔬 Tái tạo",\n                type="primary",\n                use_container_width=True\n            ):'
        ),
        (
            r'st\.caption\(f"Shape: {sinogram\.shape\[0\]} angles × {sinogram\.shape\[1\]} detectors"\)',
            'st.caption(\n        f"Shape: {sinogram.shape[0]} angles × "\n        f"{sinogram.shape[1]} detectors"\n    )'
        ),
        (
            r'colormap = st\.selectbox\("Colormap:", \["gray", "bone", "hot"\], index=1\)',
            'colormap = st.selectbox(\n                "Colormap:",\n                ["gray", "bone", "hot"],\n                index=1\n            )'
        ),
        (
            r'ax\.set_title\(f"Reconstructed Image \({method}\)", fontsize=14, fontweight="bold"\)',
            'ax.set_title(\n            f"Reconstructed Image ({method})",\n            fontsize=14,\n            fontweight="bold"\n        )'
        ),
        (
            r'phantom = resize\(phantom, reconstructed\.shape, anti_aliasing=True\)',
            'phantom = resize(\n                    phantom,\n                    reconstructed.shape,\n                    anti_aliasing=True\n                )'
        ),
        (
            r'col3\.metric\("Max Error", f"{np\.max\(np\.abs\(phantom - reconstructed\)\):\.4f}"\)',
            'col3.metric(\n                "Max Error",\n                f"{np.max(np.abs(phantom - reconstructed)):.4f}"\n            )'
        ),
        (
            r'"💡 Tip: Use Shepp-Logan phantom to test different reconstruction parameters"',
            '"💡 Tip: Use Shepp-Logan phantom to test different "\n    "reconstruction parameters"'
        ),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed: {file_path.name}")
        return True
    else:
        print(f"⏭️ No changes: {file_path.name}")
        return False


if __name__ == "__main__":
    pages_dir = Path(__file__).parent / "pages"
    fixed_count = 0
    
    for file_path in pages_dir.glob("*.py"):
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} files")
