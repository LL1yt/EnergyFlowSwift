#!/usr/bin/env python3
"""
Simple script to replace Unicode emojis with ASCII equivalents in Python files.
Used to fix Windows console encoding issues.
"""

import os
import re

# Mapping of Unicode emojis to ASCII equivalents
EMOJI_REPLACEMENTS = {
    "[OK]": "[OK]",
    "[START]": "[START]",
    "[WARN]": "[WARN]",
    "[ERROR]": "[ERROR]",
    "[TOOL]": "[TOOL]",
    "[DATA]": "[DATA]",
    "[IDEA]": "[IDEA]",
    "[TARGET]": "[TARGET]",
    "[NEW]": "[NEW]",
    "[BUG]": "[BUG]",
    "[HOT]": "[HOT]",
    "[STAR]": "[STAR]",
    "[SHINE]": "[SHINE]",
    "[UP]": "[UP]",
    "[DOWN]": "[DOWN]",
    "[SYNC]": "[SYNC]",
    "[WIN]": "[WIN]",
    "[PARTY]": "[PARTY]",
    "[STRONG]": "[STRONG]",
    "[ALERT]": "[ALERT]",
    "[NOTE]": "[NOTE]",
    "[SEARCH]": "[SEARCH]",
    "[FAST]": "[FAST]",
    "[BUILD]": "[BUILD]",
    "[ART]": "[ART]",
    "[TEST]": "[TEST]",
    "[SCIENCE]": "[SCIENCE]",
    "[LEARN]": "[LEARN]",
    "[COMPUTER]": "[COMPUTER]",
    "[DESKTOP]": "[DESKTOP]",
    "[PHONE]": "[PHONE]",
    "[PRINTER]": "[PRINTER]",
    "[FOLDER]": "[FOLDER]",
    "[DIRECTORY]": "[DIRECTORY]",
    "[FILE]": "[FILE]",
    "[CALENDAR]": "[CALENDAR]",
    "[TIME]": "[TIME]",
    "[ALARM]": "[ALARM]",
    "[BELL]": "[BELL]",
    "[LOCK]": "[LOCK]",
    "[UNLOCK]": "[UNLOCK]",
    "[STOP]": "[STOP]",
    "[CLEAN]": "[CLEAN]",
    "[WASH]": "[WASH]",
    "[SPRAY]": "[SPRAY]",
    "[TEXT]": "[TEXT]",
    "[NUMBER]": "[NUMBER]",
    "[SYMBOL]": "[SYMBOL]",
    "[ARCHIVE]": "[ARCHIVE]",
    "[RULER]": "[RULER]",
    "[RACE]": "[RACE]",
    "[START]": "[START]",
    "[FACTORY]": "[FACTORY]",
    "[HOME]": "[HOME]",
    "[HOUSE]": "[HOUSE]",
    "[SCHOOL]": "[SCHOOL]",
    "[OFFICE]": "[OFFICE]",
    "[HOSPITAL]": "[HOSPITAL]",
    "[BANK]": "[BANK]",
    "[DISK]": "[DISK]",
    "[CD]": "[CD]",
    "[DISK]": "[DISK]",
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
    """Main function to process all Python files in the project."""
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
            if fix_file(file):
                fixed_count += 1
    
    # Check subdirectories
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        total_count += 1
                        if fix_file(filepath):
                            fixed_count += 1

    print(f"\nProcessed {total_count} files, fixed {fixed_count} files")


if __name__ == "__main__":
    main()
