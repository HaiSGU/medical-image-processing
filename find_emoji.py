import re# Remove emoji manually

import sys

files = ['pages/1_Anonymization.py', 'pages/2_Segmentation.py', 'pages/3_CT_Reconstruction.py', 'pages/4_MRI_Reconstruction.py', 'pages/5_Preprocessing.py']

f = "pages/1_Anonymization.py"

emoji_pattern = re.compile("[\U0001F300-\U0001F9FF\u2600-\u27BF]")with open(f, "r", encoding="utf-8") as file:

    lines = file.readlines()

for filepath in files:

    with open(filepath, 'r', encoding='utf-8') as f:count = 0

        lines = f.readlines()for i, line in enumerate(lines):

        if "📤" in line or "✅" in line or "⚠️" in line or "❌" in line or "💡" in line:

    found = False        print(f"Line {i+1}: {line.strip()}")

    for i, line in enumerate(lines, 1):        count += 1

        if emoji_pattern.search(line):

            print(f"{filepath}:{i}: {line.rstrip()}")print(f"\nFound {count} lines with emoji")

            found = True
    
    if not found:
        print(f"{filepath}: ✓ No emoji")
