import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict
import cv2



def file_number_extract(filepath):
    filename = os.path.basename(filepath)
    number = int(filename.split('_')[-1].split('.')[0])
    return number

def fatigue_calc(files):
    """
    Calculates fatigue damage parameter. Returns dict. of {subset_id:[corresponding fatigue damage parameters, all files, 
    in order of file number]}.

    Parameters:
    -----------
    files : list
        List of str filepaths. Filepaths are of dice output .txt files.
    """
    filepaths_sorted = sorted(files, key=file_number_extract)

    subset_dict = {}

    for filepath in filepaths_sorted:
        try:
            df = pd.read_csv(filepath, skiprows=1, header=None)
            df.columns = [
                'SUBSET_ID', 'COORDINATE_X', 'COORDINATE_Y', 'DISPLACEMENT_X', 'DISPLACEMENT_Y',
                'SIGMA', 'GAMMA', 'BETA', 'STATUS_FLAG', 'UNCERTAINTY',
                'VSG_STRAIN_XX', 'VSG_STRAIN_YY', 'VSG_STRAIN_XY'
            ]

            strain_xx = pd.to_numeric(df['VSG_STRAIN_XX'], errors='coerce')
            strain_yy = pd.to_numeric(df['VSG_STRAIN_YY'], errors='coerce')
            strain_xy = pd.to_numeric(df['VSG_STRAIN_XY'], errors='coerce')

            # Calculate principal strain

            epsilon_max = (strain_xx+strain_yy)/2+(((strain_xx-strain_yy)/2)**2+(strain_xy/2)**2)**(1/2)
            epsilon_min = (strain_xx+strain_yy)/2-(((strain_xx-strain_yy)/2)**2+(strain_xy/2)**2)**(1/2)

            # Calculate fatigue damage parameter

            y_axis_calc = ((epsilon_max-epsilon_min)/2)
            
            # Adds to dict with subset ids and corresponding damage parameter lists
            for subset_id,fatigue_damage_param in y_axis_calc.items():
                if subset_id not in subset_dict:
                    subset_dict[subset_id] = []
                subset_dict[subset_id].append(fatigue_damage_param)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return subset_dict

def calc_crit_subset(original_names_text_file, dice_files):
    """
    Calculates a critical subset. Also returns dictionary of the form {cycle # : list of associated dice result file paths (.txt paths)}.

    Parameters:
    -----------
    original_names_text_file : str
        Path to the .txt file containing image original names.
    dice_files: list
        List containing str of dice output file paths (.txt paths).
    """

    cycle_to_paths = create_cycle_to_filepath_dict(original_names_text_file, dice_files)
    all_subset_deltas = defaultdict(list)

    for cycle_num, file_list in cycle_to_paths.items():
        subset_dict = fatigue_calc(file_list)

        for subset_id, values in subset_dict.items():
            clean_vals = pd.Series(values).dropna()

            # IQR Filtering
            Q1 = clean_vals.quantile(0.25)
            Q3 = clean_vals.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            filtered = clean_vals[(clean_vals >= lower) & (clean_vals <= upper)]

            if not filtered.empty:
                delt_e = filtered.max() - filtered.min()
                all_subset_deltas[subset_id].append(delt_e)

    # Aggregate across all cycles
    avg_deltas = {subset_id: sum(deltas) / len(deltas)
                  for subset_id, deltas in all_subset_deltas.items() if deltas}

    # Pick the final critical subset
    if not avg_deltas:
        raise ValueError("No valid subset found after filtering.")

    critical_subset = max(avg_deltas, key=avg_deltas.get)
    return critical_subset, cycle_to_paths

def create_cycle_to_filepath_dict(original_names_text_file, dice_files):
    """
    Creates dictionary of the form {cycle # : list of associated dice result file paths (.txt paths)}. Matches cycle to dice
    results paths.

    Parameters:
    -----------
    original_names_text_file : str
        Path to the .txt file containing image original names.
    dice_files: list
        List containing str of dice output file paths (.txt paths).
    """
    cycle_to_files = {}

    with open(original_names_text_file, "r") as f:
        for line in f:    
            filename = os.path.basename(line.strip())
            parts = filename.split('_')
            file_number = int(filename.split('_')[4].split('.')[0])
            cycle_number = int(filename.split('_')[3])
            
            if cycle_number not in cycle_to_files:
               cycle_to_files[cycle_number] = [] 
            cycle_to_files[cycle_number].append(file_number)

        for file_list in cycle_to_files.values():
            file_list.sort()

        dice_file_map = {file_number_extract(path):path for path in dice_files}

        cycle_to_paths = {}

        for cycle_num, file_nums in cycle_to_files.items():

            sorted_paths = []
            for num in sorted(file_nums):
                if num in dice_file_map:
                    sorted_paths.append(dice_file_map[num])
            if sorted_paths:
                cycle_to_paths[cycle_num] = sorted_paths
    sorted_items = sorted(cycle_to_paths.items())
    sorted_dict = dict(sorted_items)

    return sorted_dict

def calc_avg_delt_e(original_names_text_file):
    """
    Calculate average delta epsilon over provided cycles and plots strain vs. time.

    Parameters:
    -----------
    original_names_text_file : str
        Path to the .txt file containing image original names.
    """
    all_fatigue_values = []
    all_subset_ids = []
    x_vals = []
    all_delt_e_vals = []

    # Track cycle boundaries for x-axis tick labeling
    cycle_boundaries = {}

    # Get final critical subset and full mapping of cycles to file paths
    final_crit_subset, cycle_num_to_paths_dict = calc_crit_subset(original_names_text_file, dice_files)

    x_pos = 0  # Running x-axis position (across files)

    for cycle_num, file_list in cycle_num_to_paths_dict.items():
        cycle_boundaries[cycle_num] = x_pos + 1

        subset_fatigue_dict = fatigue_calc(file_list)

        for subset_id, values in subset_fatigue_dict.items():
            n_points = len(values)
            x_range = list(range(x_pos + 1, x_pos + 1 + n_points))

            all_fatigue_values.append(values)
            all_subset_ids.append(subset_id)
            x_vals.append(x_range)

            if subset_id == final_crit_subset:
                delt_e = max(values) - min(values)
                all_delt_e_vals.append(delt_e)

        x_pos += len(file_list)

    avg_delt_e = sum(all_delt_e_vals) / len(all_delt_e_vals)

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot non-critical subsets first (faint blue)
    for subset_id, y_vals, x in zip(all_subset_ids, all_fatigue_values, x_vals):
        if subset_id != final_crit_subset:
            plt.plot(x, y_vals, color='blue', alpha=0.3)

    # Plot critical subset last (bold red, on top)
    for subset_id, y_vals, x in zip(all_subset_ids, all_fatigue_values, x_vals):
        if subset_id == final_crit_subset:
            plt.plot(x, y_vals, color='red', linewidth=2, label=f'Critical Subset')

    tick_positions = list(cycle_boundaries.values())
    tick_labels = list(cycle_boundaries.keys())
    plt.xticks(tick_positions, tick_labels, fontname='Arial')
    plt.yticks(fontname='Arial')
    plt.xlabel('Cycle number', fontsize=12, fontname= 'Arial')
    plt.ylabel('Fatigue damage\nparameter (Δε)', rotation=0, fontsize=12, labelpad=50, fontname = 'Arial')
    plt.title('Fatigue Damage vs. Cycle Number', fontsize=16, fontname = 'Arial')
    plt.ylim(0, 0.05)  # Limit y-axis
    plt.tight_layout()

    # Only label the critical subset
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()

    return avg_delt_e

def show_crit_loc(crit_subset_id, results_folder, images_folder):
    """
    Plot the location of the critical subset on the first image of the images folder.

    Parameters:
    -----------
    crit_subset_id : int
        ID of the critical subset to plot.
    results_folder : str
        Path to the 'results' folder (contains DICe .txt output files).
    images_folder : str
        Path to the 'images' folder (contains raw images used in DICe).
    """

    # --- Step 1: Get and load first image ---
    image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
    if not image_files:
        raise FileNotFoundError("No image files found in images folder.")
    first_image_path = os.path.join(images_folder, image_files[0])
    img = cv2.imread(first_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) → RGB (matplotlib)

    # --- Step 2: Get coordinates of critical subset ---
    # Pick the first .txt result file from results folder
    dice_files = sorted([f for f in os.listdir(results_folder) if f.endswith('.txt')])
    if not dice_files:
        raise FileNotFoundError("No DICe .txt files found in results folder.")
    first_result_file = os.path.join(results_folder, dice_files[0])

    # Read dataframe
    df = pd.read_csv(first_result_file, skiprows=1, header=None)
    df.columns = [
        'SUBSET_ID', 'COORDINATE_X', 'COORDINATE_Y', 'DISPLACEMENT_X', 'DISPLACEMENT_Y',
        'SIGMA', 'GAMMA', 'BETA', 'STATUS_FLAG', 'UNCERTAINTY',
        'VSG_STRAIN_XX', 'VSG_STRAIN_YY', 'VSG_STRAIN_XY'
    ]

    subset_row = df[df['SUBSET_ID'] == crit_subset_id]
    if subset_row.empty:
        raise ValueError(f"Critical subset {crit_subset_id} not found in results file.")

    x_coord = subset_row['COORDINATE_X'].values[0]
    y_coord = subset_row['COORDINATE_Y'].values[0]

    # --- Step 3: Plot image with critical subset marker ---
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.scatter(x_coord, y_coord, c='red', s=80, marker='.', label=f'Critical Subset {crit_subset_id}')
    plt.legend()
    plt.title("Critical Subset Location on First Image")
    plt.axis('off')
    plt.show()


# MAIN

# USER- enter path to folder with ALL notch DICe directories for this specimen
all_notches_folder = 'path/to/dice/directories/for/all/notches'

for notch_name in os.listdir(all_notches_folder):
    notch_path = os.path.join(all_notches_folder, notch_name)
    if os.path.isdir(notch_path) and notch_name.startswith("notch_"):
        current_notch_num = int(notch_name.split("_")[1])

        # USER - Enter the pattern of the .txt file with the original filenames (current notch is a changing variable)
        original_names_text_file = f'174_H1050_cycles100to110_sorted_Notch_0{current_notch_num}_original_filenames.txt'

        # USER - Enter the pattern of the sorted notch images subfolders (current notch is a changing variable)
        images_folder = f'/Users/janhavijoglekar/Desktop/174_H1050_cycles100to110_sorted/Notch_0{current_notch_num}' 
        
        results_folder_path = os.path.join(notch_path, "results")
        if os.path.isdir(results_folder_path):
            print("Found results folder:", results_folder_path)

            # Get all .txt files inside results folder
            file_pattern = os.path.join(results_folder_path, 'DICe_solution_*.txt')
            dice_files = glob.glob(file_pattern)
            crit_sub, cycle_to_paths = calc_crit_subset(original_names_text_file, dice_files)
            delt_e = calc_avg_delt_e(original_names_text_file)

            show_crit_loc(crit_sub, results_folder_path, images_folder)

            print(f"Delta Epsilon / 2 = {delt_e/2}")
            print(f"Critical subset = {crit_sub}")

