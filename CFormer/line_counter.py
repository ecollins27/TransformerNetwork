import os

def count_non_blank_lines(filepath):
    try:
        with open(filepath, 'r', errors='ignore') as file:
            return sum(1 for line in file if line.strip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def count_lines_in_project(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.h', '.cpp', '.inl')):
                path = os.path.join(root, file)
                lines = count_non_blank_lines(path)
                print(f"{path}: {lines} non-blank lines")
                total_lines += lines
    print(f"\nTotal non-blank lines of code: {total_lines}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python count_lines.py <directory>")
    else:
        count_lines_in_project(sys.argv[1])
