#!/usr/bin/env python3
"""
Simple script to replace Unicode emojis with ASCII equivalents in Python files.
Used to fix Windows console encoding issues.
"""

import os
import re

# Mapping to restore emojis from ASCII replacements
EMOJI_REPLACEMENTS = {
    "✅": "✅",
    "🚀": "🚀", 
    "⚠️": "⚠️",
    "❌": "❌",
    "🔧": "🔧",
    "📊": "📊",
    "💡": "💡",
    "🎯": "🎯",
    "✨": "✨",
    "🐛": "🐛",
    "🔥": "🔥",
    "⭐": "⭐",
    "🌟": "🌟",
    "📈": "📈",
    "📉": "📉",
    "🔄": "🔄",
    "🏆": "🏆",
    "🎉": "🎉",
    "💪": "💪",
    "🚨": "🚨",
    "📝": "📝",
    "🔍": "🔍",
    "⚡": "⚡",
    "🛠️": "🛠️",
    "🎨": "🎨",
    "🧪": "🧪",
    "🔬": "🔬",
    "📚": "📚",
    "💻": "💻",
    "🖥️": "🖥️",
    "📱": "📱",
    "🖨️": "🖨️",
    "🗂️": "🗂️",
    "📂": "📂",
    "📁": "📁",
    "📅": "📅",
    "⏳": "⏳",
    "⏰": "⏰",
    "🔔": "🔔",
    "🔒": "🔒",
    "🔓": "🔓",
    "🛑": "🛑",
    "🧹": "🧹",
    "🧼": "🧼",
    "🧽": "🧽",
    "🔤": "🔤",
    "🔢": "🔢",
    "🔣": "🔣",
    "🗄️": "🗄️",
    "📏": "📏",
    "🏎️": "🏎️",
    "🏭": "🏭",
    "🏠": "🏠",
    "🏡": "🏡",
    "🏫": "🏫",
    "🏢": "🏢",
    "🏥": "🏥",
    "🏦": "🏦",
    "💾": "💾",
    "💿": "💿",
}


def restore_emojis_in_file(filepath):
    """Restore emojis from ASCII replacements in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Restore emojis from ASCII
        for ascii_replacement, emoji in EMOJI_REPLACEMENTS.items():
            content = content.replace(ascii_replacement, emoji)

        # Only write if changed
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Restored emojis in: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to restore emojis in all Python files."""
    # Directories to check
    dirs_to_check = [
        "new_rebuild",
        "core", 
        "data",
        "training",
        "inference",
        "utils",
        "emergent_training",
        "dynamic_training",
        "production_training",
        "smart_resume_training"
    ]
    
    fixed_count = 0
    total_count = 0
    
    # Also check root level Python files
    for file in os.listdir("."):
        if file.endswith(".py"):
            total_count += 1
            if restore_emojis_in_file(file):
                fixed_count += 1
    
    # Check subdirectories
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        total_count += 1
                        if restore_emojis_in_file(filepath):
                            fixed_count += 1

    print(f"\n🎉 Processed {total_count} files, restored emojis in {fixed_count} files!")


if __name__ == "__main__":
    main()
