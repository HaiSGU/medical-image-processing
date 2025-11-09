"""
Script xóa emoji an toàn hơn - chỉ xóa ở đầu string
"""

import re
from pathlib import Path


def remove_leading_emoji(text):
    """Xóa emoji ở đầu chuỗi nhưng giữ nguyên phần còn lại."""
    # Match emoji ở đầu + space
    emoji_pattern = re.compile(
        r"^([\U0001F300-\U0001F9FF\U0001F600-\U0001F64F\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]+)\s*"
    )
    return emoji_pattern.sub("", text)


def fix_streamlit_calls(content):
    """Fix các st.xxx() calls để xóa emoji."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Tìm các pattern st.xxx("emoji text")
        if "st.title" in line or "st.header" in line or "st.subheader" in line:
            # Extract string trong ngoặc kép
            match = re.search(r'(st\.\w+\()(["\'])(.*?)\2', line)
            if match:
                prefix = match.group(1)
                quote = match.group(2)
                text = match.group(3)
                cleaned = remove_leading_emoji(text)
                # Rebuild line
                line = (
                    line[: match.start()]
                    + f"{prefix}{quote}{cleaned}{quote}"
                    + line[match.end() :]
                )

        # Tương tự với button, success, info, warning, error
        elif any(
            x in line
            for x in ["st.button", "st.success", "st.info", "st.warning", "st.error"]
        ):
            match = re.search(r'(st\.\w+\()(["\'])(.*?)\2', line)
            if match:
                prefix = match.group(1)
                quote = match.group(2)
                text = match.group(3)
                cleaned = remove_leading_emoji(text)
                line = (
                    line[: match.start()]
                    + f"{prefix}{quote}{cleaned}{quote}"
                    + line[match.end() :]
                )

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(file_path):
    """Process một file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    content = fix_streamlit_calls(content)

    if content != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Fixed: {file_path.name}")
        return True
    else:
        print(f"- No change: {file_path.name}")
        return False


if __name__ == "__main__":
    pages_dir = Path(__file__).parent / "pages"
    fixed = 0

    for file_path in sorted(pages_dir.glob("*.py")):
        if process_file(file_path):
            fixed += 1

    print(f"\n✓ Fixed {fixed} files")
