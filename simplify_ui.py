"""
Remove emoji and simplify UI in all page files
"""

import re
from pathlib import Path

pages_dir = Path(r"d:\Documents\Medical Image Processing\pages")

# Replacements to make
replacements = [
    # Remove emojis from common UI elements
    (r'st\.title\("([🔒🧠🔬🧲⚙️📁📤📥📄📊📈🖼️💾📝🎓👁️]+)\s*', r'st.title("'),
    (r'st\.header\("([🔒🧠🔬🧲⚙️📁📤📥📄📊📈🖼️💾📝🎓👁️]+)\s*', r'st.header("'),
    (r'st\.subheader\("([🔒🧠🔬🧲⚙️📁📤📥📄📊📈🖼️💾📝🎓👁️]+)\s*', r'st.subheader("'),
    # Remove emoji from expandersst.expander
    (r'st\.expander\("ℹ️\s*', r'st.expander("'),
    (r'st\.expander\("💡\s*', r'st.expander("'),
    (r'st\.expander\("📖\s*', r'st.expander("'),
    # Remove emoji from info/warning/success boxes
    (r'st\.info\("💡\s*', r'st.info("'),
    (r'st\.success\("✅\s*', r'st.success("'),
    (r'st\.warning\("⚠️\s*', r'st.warning("'),
    (r'st\.error\("❌\s*', r'st.error("'),
    # Remove emoji from spinners
    (r'st\.spinner\("🔄\s*', r'st.spinner("'),
    (r'st\.spinner\("🧠\s*', r'st.spinner("'),
    (r'st\.spinner\("⏳\s*', r'st.spinner("'),
    # Remove page_icon from set_page_config
    (r'page_icon="[^"]+",?\s*', r""),
]

# Process all Python files in pages/
for file_path in pages_dir.glob("*.py"):
    print(f"Processing: {file_path.name}")

    # Read content
    content = file_path.read_text(encoding="utf-8")
    original = content

    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write back if changed
    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Updated: {file_path.name}")
    else:
        print(f"  - No changes: {file_path.name}")

print("\nDone!")
