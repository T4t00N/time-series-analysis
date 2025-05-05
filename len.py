def count_lines_and_validate_columns(file_path, expected_columns=10):
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            columns = line.strip().split()
            if len(columns) != expected_columns:
                print(f"Warning: Line {i} has {len(columns)} columns (expected {expected_columns})")
            line_count += 1

    print(f"Total number of lines (rows): {line_count}")
    return line_count

# Example usage
if __name__ == "__main__":
    path_to_asc = r"data\Measurement2.asc"  # Replace with your actual path
    count_lines_and_validate_columns(path_to_asc)
