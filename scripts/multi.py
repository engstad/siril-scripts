# multi.py
# Author: Pål-Kristian Engstad
#

import os
import glob
import re
from astropy.io import fits
import sirilpy as srl
import numpy as np

# --- HELPER FUNCTIONS ---
def get_exposure_time(fits_path):
    """Reads EXPTIME from FITS header or extracts it from the filename string."""
    try:
        with fits.open(fits_path) as hdul:
            if 'EXPTIME' in hdul.header:
                return float(hdul.header['EXPTIME'])
    except Exception:
        pass
    
    # Fallback to parse filename (e.g. result_1200s.fit)
    base = os.path.basename(fits_path)
    if 's.fit' in base:
        try:
            return float(base.split('_')[-1].replace('s.fit', ''))
        except ValueError:
            pass
    raise KeyError(f"Could not determine exposure time for {fits_path}")

def get_image_statistics_via_astropy(fits_path):
    """Natively computes Median and StdDev from a central 200x200 box in Python."""
    with fits.open(fits_path) as hdul:
        # Get the pixel data array (usually HDU 0 or 1 depending on Siril metadata tracking)
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        
        # If it's a 3D array (RGB color image), process it in 2D or take the first channel
        if len(data.shape) == 3:
            data = data[0] # Sample the first channel (Red/Ha) to determine background levels
            
        height, width = data.shape
        
        # Statistical Sigma-Clipping to reject nebula and stars globally
        # Step 1: Downsample by a factor of 10 to process memory instantly
        flat_data = data[::10, ::10].flatten()
        
        # Step 2: Iteratively reject pixels brighter than 3 standard deviations
        for _ in range(3):
            med = np.median(flat_data)
            std = np.std(flat_data)
            flat_data = flat_data[flat_data < (med + 3 * std)]
        
        median_val = float(np.median(flat_data))
        sigma_val = float(np.std(flat_data))
         
        # Fallback if standard deviation is zero to avoid division errors in Pixel Math
        if sigma_val == 0.0:
            sigma_val = 0.001
            
        return median_val, sigma_val


# --- 1. INITIALIZE SIRIL & FETCH WORKING DIRECTORY ---
siril = srl.SirilInterface()
try:
    siril.connect()
except Exception as e:
    print(f"Connection update note: {e}")

working_dir = os.getcwd()

print(f"Connected to Siril. Active working directory detected at:\n{working_dir}\n")

# --- 2. REPLICATE SIRIL'S ALPHABETICAL IMPORT ORDER ---
# Parse Siril's conversion text log directly to link registered frames to source files
conversion_log_path = os.path.join(working_dir, "aligned_conversion.txt")
if not os.path.exists(conversion_log_path):
    print(f"Error: Missing conversion log map at {conversion_log_path}")
    exit()

print("Parsing aligned_conversion.txt to map channel metrics smoothly...")

channel_groups = {}

with open(conversion_log_path, 'r') as f:
    for line in f:
        # Match paths inside single quotes: 'source_path' -> 'aligned_XXXXX.fit'
        matches = re.findall(r"'(.*?)'", line)
        if len(matches) != 2:
            continue
        
        src_path, aligned_basename = matches[0], matches[1]
        src_filename = os.path.basename(src_path)
        
        # Identify group and exposure properties from the real underlying source files
        prefix_match = re.match(r"^(ho|rgb|so)-", src_filename)
        if not prefix_match:
            continue
        group_key = prefix_match.group(1)
        
        real_target_path = os.path.realpath(os.path.join(working_dir, src_filename))
        try:
            exp_time = get_exposure_time(real_target_path)
        except KeyError:
            print(f"Skipping {src_filename}: Exposure metadata check failed.")
            continue
            
        # Map to your registration prefix filename (r_aligned_XXXXX.fit)
        aligned_num_str = aligned_basename.replace("aligned_", "").replace(".fit", "")
        aligned_filename = f"r_aligned_{aligned_num_str}.fit"
        aligned_path = os.path.abspath(os.path.join(working_dir, aligned_filename))
     
        if not os.path.exists(aligned_path):
            print(f"Warning: Expected registered file missing: {aligned_filename}. Skipping slot.")
            continue

        if group_key not in channel_groups:
            channel_groups[group_key] = []
        
        channel_groups[group_key].append({
            'path': aligned_path,
            'filename_no_ext': aligned_filename.replace(".fit", ""),
            'exp': exp_time,
            'idx': int(aligned_num_str),
            'src': src_filename
        })

# Print discovery log
print("\n--- Map Matrix ---")
for group, items in channel_groups.items():
    print(f"Channel Group [{group.upper()}]:")
    for item in items:
        print(f"  -> Slot #{item['idx']}: {os.path.basename(item['path'])} matched to Source {item['src']} | Exposure: {item['exp']}s")

for group, targets in channel_groups.items():
    if len(targets) < 2:
        print(f"\nSkipping group {group.upper()}: Insufficient session files to stack.")
        continue
        
    print(f"\n==========================================")
    print(f"Processing Dynamic Stack: {group.upper()}")
    print(f"==========================================")
    
    # Sort targets explicitly by index slot order
    targets = sorted(targets, key=lambda x: x['idx'])
    total_group_time = sum(t['exp'] for t in targets)

    # Define the output master path explicitly BEFORE running the calculations
    output_master_file = os.path.abspath(os.path.join(working_dir, f"master-{group.lower()}.fit"))
    if os.path.exists(output_master_file):
        os.remove(output_master_file)
    
    # Calculate all statistical metrics upfront globally via Astropy
    for t in targets:
        t['b'], t['s'] = get_image_statistics_via_astropy(t['path'])
        t['weight'] = t['exp'] / total_group_time

    # Compute a true global statistical baseline weighted across all nights
    b_ref = sum(t['b'] * t['weight'] for t in targets)
    s_ref = sum(t['s'] * t['weight'] for t in targets)
        
    print(f"--> Normalizing {len(targets)} frames against baseline reference (Median: {b_ref:.6f})")
    
    formula_parts = []
    for t in targets:
        normalized_img = f"(((${t['filename_no_ext']}$ - {t['b']}) / {t['s']}) * {s_ref} + {b_ref})"
        formula_parts.append(f"({normalized_img} * {t['weight']})")
    
    pixelmath_formula = " + ".join(formula_parts)
    output_master_file = os.path.abspath(os.path.join(working_dir, f"master-{group.lower()}.fit"))
    
    print(f"  -> Injecting global balanced blend formula into Siril engine...")
    siril.cmd(f'pm "{pixelmath_formula}"')
    siril.cmd(f'save "{output_master_file}"')
    print(f"✨ Created: {os.path.basename(output_master_file)} ({total_group_time/3600:.2f} total hours)")

print("\n🎉 Stacking Engine Complete!")
