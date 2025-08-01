import numpy as np
import tifffile
import h5py
import tkinter as tk
from tkinter import filedialog
import os

# Set initial directory
initial_dir = r'K:\SPT_2023\Revision\Comparison under experimental challenging condition'

# Select MAT files
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select .mat files containing 'ims' or 'timelapsedata'",
    filetypes=[("MAT files", "*.mat")],
    initialdir=initial_dir
)

# Output directory
if not file_paths:
    print("No files selected.")
else:
    output_dir = os.path.join(os.path.dirname(file_paths[0]), "tiff_output")
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        try:
            f = h5py.File(file_path, 'r')

            # Try loading variable
            if 'ims' in f:
                data = np.array(f['ims'])
            elif 'timelapsedata' in f:
                data = np.array(f['timelapsedata'])
            else:
                print(f"File {file_path} does not contain 'ims' or 'timelapsedata'")
                continue

            # Convert type if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Save depending on dimensionality
            if data.ndim == 3:
                # (Y, X, T) -> (T, Y, X)
                ims = data
                output_name = os.path.basename(file_path).replace('.mat', '.tif')
                output_path = os.path.join(output_dir, output_name)
                tifffile.imwrite(output_path, ims, imagej=True, metadata={'axes': 'TYX'})
                print(f"Saved 3D: {output_path}")
            elif data.ndim == 4:
                N = data.shape[0]
                for i in range(N):
                    substack = data[i, :, :, :]  # shape: (File, T, X, Y)
                    # substack = np.transpose(substack, (1, 2, 0))  # (T, Y, X)
                    base_name = os.path.basename(file_path).replace('.mat', f'_{i:03d}.tif')
                    output_path = os.path.join(output_dir, base_name)
                    tifffile.imwrite(output_path, substack, imagej=True, metadata={'axes': 'TYX'})
                print(f"Saved 4D: {N} files from {file_path}")
            else:
                print(f"Unsupported data shape {data.shape} in file {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
