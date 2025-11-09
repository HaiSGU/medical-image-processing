#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove all emoji from pages"""

import os

# All emoji to remove
EMOJI_LIST = [
    "📤",
    "🧠",
    "🔬",
    "🧲",
    "🖼️",
    "✅",
    "❌",
    "⚠️",
    "💡",
    "📊",
    "🎯",
    "📈",
    "🔍",
    "⚙️",
    "📝",
    "🎨",
    "🔧",
    "📁",
    "💾",
    "🚀",
]

files = [
    "pages/1_Anonymization.py",
    "pages/2_Segmentation.py",
    "pages/3_CT_Reconstruction.py",
    "pages/4_MRI_Reconstruction.py",
    "pages/5_Preprocessing.py",
]

print("Đang xóa emoji...")
for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for emoji in EMOJI_LIST:
        content = content.replace(emoji + " ", "")  # emoji + space
        content = content.replace(emoji, "")  # emoji alone

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ {filepath}")
    else:
        print(f"- {filepath} (không có emoji)")

print("\n✓ Hoàn tất!")
