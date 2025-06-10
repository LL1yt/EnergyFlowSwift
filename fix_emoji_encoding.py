#!/usr/bin/env python3
"""
Script to replace Unicode emoji with ASCII symbols for Windows compatibility
Скрипт для замены Unicode эмодзи на ASCII символы для совместимости с Windows
"""

import os
import re
from pathlib import Path
import argparse

# Mapping of emoji to ASCII replacements
EMOJI_REPLACEMENTS = {
    # Warning and status symbols
    "[WARNING]": "[WARNING]",
    "[OK]": "[OK]",
    "[ERROR]": "[ERROR]",
    "[START]": "[START]",
    "[TARGET]": "[TARGET]",
    "[DATA]": "[DATA]",
    "[INFO]": "[INFO]",
    "[CONFIG]": "[CONFIG]",
    "[BOT]": "[BOT]",
    "[SUCCESS]": "[SUCCESS]",
    "[TIME]": "[TIME]",
    "[STOP]": "[STOP]",
    "[PC]": "[PC]",
    "[TEST]": "[TEST]",
    "[PLAY]": "[PLAY]",
    # Additional symbols that might appear
    "[SAVE]": "[SAVE]",
    "[FOLDER]": "[FOLDER]",
    "[FILE]": "[FILE]",
    "[MAGNIFY]": "[SEARCH]",
    "[STAR]": "[STAR]",
    "[IDEA]": "[IDEA]",
    "[HOT]": "[HOT]",
    "[FAST]": "[FAST]",
    "[WEB]": "[WEB]",
    "[PACKAGE]": "[PACKAGE]",
    "[ALERT]": "[ALERT]",
    "[CHART]": "[CHART]",
    "[COMPUTER]": "[COMPUTER]",
    "[LINK]": "[LINK]",
    "[CIRCUS]": "[CIRCUS]",
    "[ART]": "[ART]",
    "[MUSIC]": "[MUSIC]",
    "[SHINE]": "[SHINE]",
    "[FLAG]": "[FLAG]",
    "[TROPHY]": "[TROPHY]",
    "[WORK]": "[WORK]",
    "[BOOKS]": "[BOOKS]",
    "[MAGNIFY]": "[MAGNIFY]",
    "[WRITE]": "[WRITE]",
    "[GEAR]": "[GEAR]",
    "[LOCK]": "[LOCK]",
    "[UNLOCK]": "[UNLOCK]",
    "[PIN]": "[PIN]",
    "[CLIP]": "[CLIP]",
    "[MASK]": "[MASK]",
    "[MOVIE]": "[MOVIE]",
    "[MIC]": "[MIC]",
    "[HEADPHONES]": "[HEADPHONES]",
    # Additional missing emoji
    "[BRAIN]": "[BRAIN]",
    "[REFRESH]": "[REFRESH]",
    "[GRADUATE]": "[GRADUATE]",
    "[DICE]": "[DICE]",
    "[PIN]": "[PIN]",
}


def replace_emoji_in_text(text: str) -> str:
    """Replace emoji in text with ASCII equivalents"""
    result = text
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        result = result.replace(emoji, replacement)
    return result


def process_file(file_path: Path, dry_run: bool = True) -> tuple[bool, int]:
    """
    Process a single file to replace emoji
    Returns (changed, num_replacements)
    """
    try:
        # Read file with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace emoji
        new_content = replace_emoji_in_text(content)

        # Count changes
        if content != new_content:
            changes = 0
            for emoji in EMOJI_REPLACEMENTS.keys():
                changes += content.count(emoji)

            if not dry_run:
                # Write back with UTF-8 encoding
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"FIXED: {file_path} ({changes} replacements)")
            else:
                print(f"WOULD FIX: {file_path} ({changes} replacements)")

            return True, changes

        return False, 0

    except Exception as e:
        print(f"ERROR processing {file_path}: {e}")
        return False, 0


def scan_directory(directory: Path, file_pattern: str = "*.py", dry_run: bool = True):
    """Scan directory for files with emoji"""
    print(f"Scanning {directory} for {file_pattern}...")

    total_files = 0
    changed_files = 0
    total_changes = 0

    for file_path in directory.rglob(file_pattern):
        # Skip __pycache__, .git, and .venv directories
        if any(
            skip_dir in str(file_path)
            for skip_dir in ["__pycache__", ".git", ".venv", "venv", "env"]
        ):
            continue

        total_files += 1
        changed, changes = process_file(file_path, dry_run)

        if changed:
            changed_files += 1
            total_changes += changes

    print(f"\nSummary:")
    print(f"  Files scanned: {total_files}")
    print(f"  Files with emoji: {changed_files}")
    print(f"  Total replacements: {total_changes}")

    if dry_run and changed_files > 0:
        print(f"\nTo apply changes, run with --apply flag")


def main():
    parser = argparse.ArgumentParser(description="Replace emoji with ASCII symbols")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory to scan (default: current)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.py",
        help="File pattern to match (default: *.py)",
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"ERROR: Directory {directory} does not exist")
        return 1

    print(f"Emoji Encoding Fix Script")
    print(f"Directory: {directory}")
    print(f"Pattern: {args.pattern}")
    print(f"Mode: {'APPLY CHANGES' if args.apply else 'DRY RUN'}")
    print("-" * 50)

    scan_directory(directory, args.pattern, dry_run=not args.apply)

    return 0


if __name__ == "__main__":
    exit(main())
