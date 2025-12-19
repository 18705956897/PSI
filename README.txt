================================================================================
PSI (Pixel-segmenting Imaging) Project - Code and Environment Guide
================================================================================

Dear Reviewers,

Thank you for reviewing our research amid your busy schedule. Regarding the 
attached demo code, please note:

1. This is a preliminary demo version optimized to reduce computation time.
2. It allows for quick verification of the visual enhancement effect of the 
   PSI method compared with original images.
3. The demo can directly reconstruct images with obvious visual improvements.
4. All actual before-and-after enhancement images from the paper are stored 
   in the 'Results' folder.

To verify advanced reconstruction effects:
- You can use the 'QE_measurement.py' script to generate a customized QE.npz file.
- For convenience, a pre-generated 'QE Distribution Data.npz' is attached.
- Slight parameter adjustments in the code will yield optimized results.

--------------------------------------------------------------------------------
1. System Requirements
--------------------------------------------------------------------------------
- Operating System: Windows 10/11 (Recommended); Linux/macOS compatible.
- Python Version: 3.8 or higher.

--------------------------------------------------------------------------------
2. Required Packages
--------------------------------------------------------------------------------
Install all necessary dependencies by running the following command:

pip install numpy scipy scikit-image matplotlib Pillow h5py

- Linux Users: Please install 'python3-tk' first (e.g., sudo apt-get install python3-tk).

--------------------------------------------------------------------------------
3. Script Descriptions
--------------------------------------------------------------------------------
- Code.py: Main Graphical User Interface (GUI) for reconstruction and visualization.
- Light_field_fitting.py: Algorithm for 6-parameter fringe light field fitting.
- QE_measurement.py: Core algorithm for intra-pixel QE calculation and mapping.

--------------------------------------------------------------------------------
4. Data & Running Notes
--------------------------------------------------------------------------------
- File Paths: The scripts use relative paths. IT IS HIGHLY RECOMMENDED TO KEEP 
  ALL SCRIPTS AND THEIR DATA FOLDERS IN THE SAME ROOT DIRECTORY.
- Data Organization: Ensure folders like './16_frames' and './pre_C1_S2', 
  and files like 'QE Distribution Data.npz' and 'dn0p1.npz' are placed correctly.
- Launch: Run 'Code.py' to launch the GUI and follow the prompts.

--------------------------------------------------------------------------------
Thank you again for your time and consideration.

Sincerely yours,

Ze Zhang, PhD
Corresponding Author
Aerospace Information Research Institute, Chinese Academy of Sciences
Email: zhangze@aircas.ac.cn
================================================================================