import sys

def read_file(file_path):
    """
    Read the contents of a file and return a set of rows.
    """
    with open(file_path, 'r') as file:
        # Read the lines and convert them into a set
        rows = {line.strip() for line in file}
    return rows

def common_rows(file1_path, file2_path):
    """
    Check if two files have common rows.
    """
    # Read the contents of each file
    rows_file1 = read_file(file1_path)
    rows_file2 = read_file(file2_path)
    
    # Find the common rows by computing the intersection
    common_rows_set = rows_file1.intersection(rows_file2)
    
    # Return True if there are common rows, False otherwise
    return len(common_rows_set) > 0, common_rows_set

# Example usage
file1_path = sys.argv[1]
file2_path = sys.argv[2]

have_common_rows, common_rows_set = common_rows(file1_path, file2_path)

if have_common_rows:
    print("The files have common rows.")
    print("Common rows:", common_rows_set)
else:
    print("The files do not have common rows.")