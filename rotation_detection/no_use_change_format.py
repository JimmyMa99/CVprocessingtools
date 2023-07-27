import os
import glob

# Your directory here
dir_path = '/path/to/your/directory/'

# Get all txt files in the directory
file_list = glob.glob(os.path.join(dir_path, '*.txt'))

for file in file_list:
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            split_line = line.split(',')
            # Check if split_line has at least 9 elements
            if len(split_line) < 9:
                print(f"Skipping line in file {file}: {line}")
                continue
            split_line[8] = split_line[8].replace(' ', '_').rstrip()
            new_line = ','.join(split_line) + '\n'
            f.write(new_line)
