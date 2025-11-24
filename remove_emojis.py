"""
Script đơn giản để xóa emoji khỏi các file pages
"""

import re

# Danh sách files
files = [
    r"d:\Documents\Medical Image Processing\pages\1_Anonymization.py",
    r"d:\Documents\Medical Image Processing\pages\2_Segmentation.py",
    r"d:\Documents\Medical Image Processing\pages\3_CT_Reconstruction.py",
    r"d:\Documents\Medical Image Processing\pages\4_MRI_Reconstruction.py",
    r"d:\Documents\Medical Image Processing\pages\5_Preprocessing.py",
]

# Danh sách emoji cần xóa
emojis_to_remove = [
    "🔒",
    "📥",
    "📤",
    "⚙️",
    "ℹ️",
    "✅",
    "❌",
    "⏳",
    "💾",
    "📊",
    "🔬",
    "🧠",
    "🎨",
    "🔄",
    "📖",
    "💡",
    "🚀",
    "👀",
    "📏",
    "📈",
    "🖼️",
    "🧲",
    "📍",
    "🔧",
    "📂",
    "🔇",
    "📐",
    "🔢",
    "✨",
    "🌫️",
    "🔊",
    "🎯",
    "📁",
    "1️⃣",
    "2️⃣",
    "3️⃣",
    "4️⃣",
    "5️⃣",
    "6️⃣",
    "⚠️",
]

# Tạo pattern
pattern = "[" + "".join(emojis_to_remove) + "]"

for file_path in files:
    try:
        # Đọc file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Xóa emoji
        cleaned = re.sub(pattern, "", content)

        # Ghi lại
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f'OK: {file_path.split("\\\\")[-1]}')
    except Exception as e:
        print(f'ERROR: {file_path.split("\\\\")[-1]} - {e}')

print("\\nDone!")
