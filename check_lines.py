#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check specific lines in files"""

files_to_check = [
    (r"d:\Documents\Medical Image Processing\pages\4_MRI_Reconstruction.py", 83),
    (r"d:\Documents\Medical Image Processing\pages\5_Preprocessing.py", 281),
    (r"d:\Documents\Medical Image Processing\pages\3_CT_Reconstruction.py", 268),
]

for filepath, line_num in files_to_check:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"\n{filepath} line {line_num}:")
            print(repr(lines[line_num - 1]))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
