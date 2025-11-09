#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove all emoji from pages - FINAL VERSION"""

EMOJI_LIST = ["📤", "🧠", "🔬", "🧲", "🖼️", "✅", "❌", "⚠️", "💡", "📊"]

files = [
    "pages/1_Anonymization.py",
    "pages/2_Segmentation.py",
    "pages/3_CT_Reconstruction.py",
    "pages/4_MRI_Reconstruction.py",
    "pages/5_Preprocessing.py",
]

for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    for emoji in EMOJI_LIST:
        content = content.replace(emoji + " ", "")
        content = content.replace(emoji, "")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ {filepath}")

print("Done!")
