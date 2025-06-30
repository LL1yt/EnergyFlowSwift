#!/usr/bin/env python3
"""
Simple script to replace Unicode emojis with ASCII equivalents in Python files.
Used to fix Windows console encoding issues.
"""

import os
import re

# Mapping of Unicode emojis to ASCII equivalents
EMOJI_REPLACEMENTS = {
    "✅": "[OK]",
    "🚀": "[START]",
    "⚠️": "[WARN]",
    "❌": "[ERROR]",
    "🔧": "[TOOL]",
    "📊": "[DATA]",
    "💡": "[IDEA]",
    "🎯": "[TARGET]",
    "✨": "[NEW]",
    "🐛": "[BUG]",
    "🔥": "[HOT]",
    "⭐": "[STAR]",
    "🌟": "[SHINE]",
    "📈": "[UP]",
    "📉": "[DOWN]",
    "🔄": "[SYNC]",
    "🏆": "[WIN]",
    "🎉": "[PARTY]",
    "💪": "[STRONG]",
    "🚨": "[ALERT]",
    "📝": "[NOTE]",
    "🔍": "[SEARCH]",
    "⚡": "[FAST]",
    "🛠️": "[BUILD]",
    "🎨": "[ART]",
    "🧪": "[TEST]",
    "🔬": "[SCIENCE]",
    "📚": "[LEARN]",
    "💻": "[COMPUTER]",
    "🖥️": "[DESKTOP]",
    "📱": "[PHONE]",
    "🖨️": "[PRINTER]",
    "🗂️": "[FOLDER]",
    "📂": "[DIRECTORY]",
    "📁": "[FILE]",
    "📅": "[CALENDAR]",
    "⏳": "[TIME]",
    "⏰": "[ALARM]",
    "🔔": "[BELL]",
    "🔒": "[LOCK]",
    "🔓": "[UNLOCK]",
    "🛑": "[STOP]",
    "🧹": "[CLEAN]",
    "🧼": "[WASH]",
    "🧽": "[SPRAY]",
    "🔤": "[TEXT]",
    "🔢": "[NUMBER]",
    "🔣": "[SYMBOL]",
    "🗄️": "[ARCHIVE]",
    "📏": "[RULER]",
    "🏎️": "[RACE]",
    "🏁": "[START]",
    "🏭": "[FACTORY]",
    "🏠": "[HOME]",
    "🏡": "[HOUSE]",
    "🏫": "[SCHOOL]",
    "🏢": "[OFFICE]",
    "🏥": "[HOSPITAL]",
    "🏦": "[BANK]",
    "💾": "[DISK]",
    "💿": "[CD]",
    "📀": "[DISK]",
}


def fix_file(filepath):
    """Fix Unicode emojis in a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace emojis
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)

        # Only write if changed
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to process all Python files in new_rebuild directory."""
    root_dir = "new_rebuild"
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found!")
        return

    fixed_count = 0
    total_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                total_count += 1
                if fix_file(filepath):
                    fixed_count += 1

    print(f"\nProcessed {total_count} files, fixed {fixed_count} files")


if __name__ == "__main__":
    main()
