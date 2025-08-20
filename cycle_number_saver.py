import pandas as pd
import os
import glob

folder_path = '/Users/janhavijoglekar/Desktop/174_H1050_cycles100to110_sorted/Notch_08'
folder_description = folder_path.split('/')[4:6]
file_pattern = os.path.join(folder_path, '*.bmp')
files = glob.glob(file_pattern)

# Creates .txt file with original name, includes cycle #

text_file_name = f"{folder_description[0]}_{folder_description[1]}_original_filenames.txt"

with open(text_file_name, "w") as file:
    for filepath in files:
        file.write(filepath + '\n')

# Renames files in that same folder to omit the cycle #

for filepath in files:
    current_filename = os.path.basename(filepath)
    parts = current_filename.split('_')
    new_filename = '_'.join(parts[:3]+parts[4:])

    source = filepath
    dest = os.path.join(folder_path, new_filename)

    os.rename(source, dest)