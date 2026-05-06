==========================================================================================
Simulation Project (Robustness Against Non-ideal Factors) - Code and Environment Guide
==========================================================================================

Dear Reviewers,

Thank you for reviewing our research amid your busy schedule. Regarding the 
attached demo code, please note:

1. This is a preliminary demo version optimized to reduce computation time.
2. It enables rapid performance validation of the proposed evolutionary algorithm (EA) 
for QE measurement under non-ideal conditions, including Gaussian white 
noise of varying intensities, light-intensity fluctuations at different intensities, and 
phase shifts with diverse angles.
3. Gaussian white noise of varying intensities is employed to simulate thermal noise in
photosensitive devices as well as electronic chain interference in practical systems.
4. Gaussian white noise of varying intensities is added to the (fitted) light field to 
simulate light intensity fluctuations.
5. The demo can calculate the RMSE and MAE between the QE values estimated by EA and the 
ground-truth QE under the above non-ideal conditions.
6. All QE measurements obtained under non-ideal conditions are stored separately in the 
three folders titled 'QE with Gaussian noise intensities', 'QE with light-intensity 
fluctuations', and 'QE with phase shifts'.

To verify advanced QE measurement effects:
- Slight parameter adjustments in the code, such as appropriately increasing the 
population sizes and iteration numbers, will yield optimized results.

------------------------------------------------------------------------------------------
1. System Requirements
------------------------------------------------------------------------------------------
- Operating System: Windows 10/11 (Recommended); Linux/macOS compatible.
- Python Version: 3.8 or higher.

------------------------------------------------------------------------------------------
2. Required Packages
------------------------------------------------------------------------------------------
Install all necessary dependencies by running the following command:

pip install numpy scipy scikit-image matplotlib Pillow h5py

- Linux Users: Please install 'python3-tk' first (e.g., sudo apt-get install python3-tk).

------------------------------------------------------------------------------------------
3. Script Descriptions
------------------------------------------------------------------------------------------
- QE_measurement_with_Gaussian_noise_intensities.py: Algorithm for QE measurement 
under Gaussian white noise of varying intensities.
- QE_measurement_with_light-intensity_fluctuations.py: Algorithm for QE measurement
under light-intensity fluctuations at different intensities.
- QE_measurement_with_phase_shifts.py: Algorithm for QE measurement under phase 
shifts with diverse angles.

------------------------------------------------------------------------------------------
4. Data & Running Notes
------------------------------------------------------------------------------------------
- File Paths: The scripts use relative paths. IT IS HIGHLY RECOMMENDED TO KEEP 
  ALL SCRIPTS AND THEIR DATA FOLDERS IN THE SAME ROOT DIRECTORY.
- Data Organization: Ensure the folder './Raw Data' is placed correctly.
- Launch: Under different non-ideal conditions, the RMSE and MAE between the estimated 
and true QE values can be obtained by running the following scripts respectively: 
'QE_measurement_with_Gaussian_noise_intensities.py', 'QE_measurement_with_phase_shifts.py'
 and 'QE_measurement_with_light-intensity_fluctuations.py'.
------------------------------------------------------------------------------------------
Thank you again for your time and consideration.

Sincerely yours,

Ze Zhang, PhD
Corresponding Author
Aerospace Information Research Institute, Chinese Academy of Sciences
Email: zhangze@aircas.ac.cn
==========================================================================================
