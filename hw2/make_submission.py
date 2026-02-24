import zipfile
import os
import glob

def make_zip():
    patterns_to_include = [
        # cartpole
        'exp/*cartpole_sd*',
        'exp/*cartpole_rtg_sd*',
        'exp/*cartpole_na_sd*',
        'exp/*cartpole_rtg_na_sd*',
        'exp/*cartpole_lb_sd*',
        'exp/*cartpole_lb_rtg_sd*',
        'exp/*cartpole_lb_na_sd*',
        'exp/*cartpole_lb_rtg_na_sd*',
        # cheetah
        'exp/*cheetah_sd*',
        'exp/*cheetah_baseline_sd*',
        'exp/*cheetah_baseline_low_lr*', # Added to include all variations of baseline explicitly requested in previous experiment
        # inverted pendulum
        'exp/*pendulum_sd*', # Updated pattern since actual log folder is InvertedPendulum-v4_pendulum_sd1_...
        # lunar lander
        'exp/*lunar_lander_lambda0_sd*',
        'exp/*lunar_lander_lambda0.95_sd*',
        'exp/*lunar_lander_lambda0.98_sd*',
        'exp/*lunar_lander_lambda0.99_sd*',
        'exp/*lunar_lander_lambda1_sd*'
    ]
    
    dirs_to_zip = []
    
    # Resolving glob patterns
    for pattern in patterns_to_include:
        matches = glob.glob(pattern)
        print(f"Pattern {pattern} matched: {matches}")
        dirs_to_zip.extend(matches)
        
    # Deduplicate
    dirs_to_zip = list(set(dirs_to_zip))
    print(f"Total dirs to zip from exp: {len(dirs_to_zip)}")

    # Add extra required files to zip
    extra_files = ['pyproject.toml', 'uv.lock', 'README.md']
    
    with zipfile.ZipFile('submit.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add exp dirs
        for dir_path in dirs_to_zip:
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} not found.")
                continue
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = file_path # Keep natural relative path
                    zipf.write(file_path, arcname)
                    
        # 2. Add src directory completely
        for root, _, files in os.walk('src'):
            for file in files:
                if '__pycache__' not in root:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path)
                    
        # 3. Extra top level files
        for file in extra_files:
            if os.path.exists(file):
                 zipf.write(file, file)
            else:
                 print(f"Warning: Top-level file {file} not found.")
                 
    print("Created submit.zip successfully!")

if __name__ == "__main__":
    make_zip()
