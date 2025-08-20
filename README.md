# DICe_Strain_Analysis
Calculate strain range for DICe output .txt files

- Prior to running DICe, sort the notch image folder with the notch parser. For each subfolder (notch) in your sorted folder, you must run cycle_number_saver.py (changing the folder path name for your specific subfolder). This will save all the original filenames in a .txt file, and get rid of the cycle number from the actual image names. Now the images are ready for ROI generation, use Create_DICe_ROI.m to do this. Run DICe for each notch, using subset = 35, step = 3, VSG = 15. Make sure to set a new directory each time a new notch is run in DICe.

- Make sure all DICe directories (the folders containing the "results" folder) are put in one folder. There should be 8 directories (subfolders in the main folder), one for each notch, and make sure these directories follow the naming convention "notch_X".

- USER INPUTS: In the # MAIN section of the script, first set all_notch_folder equal to the path to the root folder containing all 8 notches' DICe directories. Next, set original_names_text_file equal to the pattern of the filenames of the .txt files containing the original names of the images (ex. original_names_text_file = f'SS410_sorted_Notch_0{current_notch_num}_original_filenames.txt'). Finally, set images_folder equal to the pattern of the subfolder names of the sorted images (ex. images_folder = f'/Users/JJ/Desktop/SS410_sorted/Notch_0{current_notch_num}').

- When you run the script, it will print for example, "Found results folder: /Users/janhavijoglekar/Desktop/all_SS410_directories/notch_3/results". Then it will print the critical subset ID and delta epslion/2 value for this notch. It will also graph strain vs. time, and plot the location of the critical subset, for extra verification that the critical subset makes logical sense.
