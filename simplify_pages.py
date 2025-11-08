"""
Script tự động xóa emoji và đơn giản hóa các trang Streamlit
"""

import re
from pathlib import Path


def remove_emojis(text):
    """Xóa tất cả emoji khỏi text."""
    # Pattern để match emoji
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def simplify_file(file_path):
    """Đơn giản hóa một file Python."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # Xóa emoji
    content = remove_emojis(content)

    # Thay thế các tiêu đề có emoji
    replacements = [
        # Titles
        (r'st\.title\(".*?\s+', 'st.title("'),
        (r'st\.subheader\(".*?\s+', 'st.subheader("'),
        (r'st\.header\(".*?\s+', 'st.header("'),
        # Buttons
        (r'st\.button\(".*?\s+', 'st.button("'),
        # Success/Info/Warning messages
        (r'st\.success\(".*?\s+', 'st.success("'),
        (r'st\.info\(".*?\s+', 'st.info("'),
        (r'st\.warning\(".*?\s+', 'st.warning("'),
        (r'st\.error\(".*?\s+', 'st.error("'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Lưu nếu có thay đổi
    if content != original:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Đã sửa: {file_path.name}")
        return True
    else:
        print(f"- Không đổi: {file_path.name}")
        return False


if __name__ == "__main__":
    pages_dir = Path(__file__).parent / "pages"
    fixed = 0

    for file_path in sorted(pages_dir.glob("*.py")):
        if simplify_file(file_path):
            fixed += 1

    print(f"\n✓ Đã xử lý {fixed} file")
