# SPTnet (single-particle tracking neural network)
This repository accompanies the manuscript:
“SPTnet: a deep learning framework for end-to-end single-particle tracking and motion dynamics analysis”
by Cheng Bi, Kevin L. Scrudders, Yue Zheng, Maryam Mahmoodi, Shalini T. Low-Nam, and Fang Huang.


## Repository contents
```text
Python files
├── SPTnet_toolbox.py          # SPTnet architecuture and Utilities for data loading
├── SPTnet_training.py         # Training script for SPTnet
├── SPTnet_inference.py        # Inference script for trained model
├── transformer.py             # Spatial transformer module (Vaswani, A., et al., Attention is all you need. Advances in neural information processing systems, 2017)
├── transformer3D.py           # Temporal transformer module (Modified to take 3D inputs)
└── mat_to_tiff.py             # Converts .mat videos to TIFF series

MATLAB files
├── SPTnet_trainingdata_generator.m        # GUI to generate training datasets
├── Visualize_SPTnet_Outputs_GUI.m         # GUI to visualize inference results
├── CRLB_H_D_frame.mat                     # CRLB matrix used in loss function
└── compute_CRLB_matrix.m                  # MATLAB function to calcualte the CRLB matrix used for training

Example_data
├── Example_testdata.mat                 # .mat file contains 10 test videos
├── Example_data_SA-AF647_on_SLB.tif     # Experimental data in tiff format (Tracking lipid on SLB using AF647 dye)
└── Example_256by256by100.mat            # Example of a simulated video with 256 × 256 pixels and 100 frames

Example_workflow_1 (process large videos)
├── Step1_segementation.m                             # MATLAB script to do segementation
├── Step2_SPTnet_inference.py                         # Same as "SPTnet_inference.py"
├── Step3_stitch_and_display.m                        # MATLAB script to stitch segemented files
├── Step4_chunk_connection_and_repeat_reduction.m     # Connect blocks and remove duplicate tracks
└── Segementation_user_manual.pdf                     # User manual for segementation

Others
├── Trained_models                         # Pretrained model (based on a Nikon Ti2 TIRF system, NA=1.49)
├── PSF-toolbox                            # Simulated PSF generation toolbox
├── DIPimage_2.9                           # DIPimage toolbox for MATLAB (https://diplib.org/download_2.9.html)
├── Segmentation                           # Scripts for segmentation and stitching
├── SPTnet_environment.yml                 # Conda environment configuration included all necessary packages for SPTnet
├── package_required_for_SPTnet.txt        # Packages required by SPTnet
├── Installation of SPTnet.pdf             # Installation instruction
└── SPTnet user manual.pdf                 # User manual
 ```
## 1. Installation
Python environment (via Anaconda)

**(1)**  Install Anaconda or Miniconda

**(2)** Through "Anaconda prompt" navigate to the folder with "SPTnet_environment.yml"

**(3)** Run the following command to create the environment
```
conda env create -f SPTnet_environment.yml
```
## 2. Instructions for generating training videos
This code has been tested on the following systems and packages:
Microsoft Windows 10 Education, Matlab R2021a, DIPimage 2.9 (http://www.diplib.org/)

**(1)** Change MATLAB current folder to the directory that contains “PSF-toolbox”.
Run ‘SPTnet_trainingdata_generator.m’ to generate the training dataset.

<img width="356" height="476" alt="image" src="https://github.com/user-attachments/assets/60b7c8d3-8734-48b0-a529-6ed8f554cc66" />


**(2)** The default settings will generate 5 files each containing 100 videos.

![image](https://github.com/user-attachments/assets/51f965b0-6846-447f-b568-bc67e5745a35)

**Note:** SPTnet was trained with >200,000 videos to achieve precision approching CRLB. To generate more training and validation datasets, please locate the variables specifying the number of training videos in ‘SPTnet_trainingdata_generator.m’ (e.g., Num_file, Videos_per_file). Increase these values to the desired amount and producing additional .mat files containing more simulated videos for training and validation.

## 3. Instructions for training SPTnet using simulated training datasets
The code has been tested on the following systems and packages:
Ubuntu20.04LTS, Python3.9.12, Pytorch1.11.0, CUDA11.3, MatlabR2021a

**To start training,**

**(1)** Type the following command in the terminal: 
```
python SPTnet_training.py
```
**(2)** Select the folder to save the trained model

**(3)** Select the training data files.

**(4)** During the training, the model with the minimal validation loss will be saved as ‘trained_model’ onto the selected folder in step (2), together with an image of the training loss and validation loss changes along with training epoch.

<img width="426" height="318" alt="image" src="https://github.com/user-attachments/assets/5c222473-7ae7-4c10-962d-c13f0546f5dc" />

(Example learning curves, training - red, validation - blue)


## 4. Instructions for running inference using a trained model
To test the trained model,

**(1)** Type the following command in terminal: 
```
python SPTnet_inference.py
```
**(2)** Select the trained model you will use for inference

**(3)** Select the video file that will be analyzed by SPTnet

**(4)** An output ‘.mat’ file will be generated under the ‘inference_results’ folder located in the directory of the selected model in step (2), which contains all the estimated trajectories, detection probabilities, Hurst exponents, and generalized diffusion coefficients ready for analysis or visualization.

**Note:** On a typical GPU-enabled PC, SPTnet can process a 30-frame video (each frame sized 64×64 pixels) in approximately 60 ms. Actual performance may vary depending on specific hardware configurations (GPU model, CPU, memory, etc.)

## 5. Instructions for visualizing the SPTnet output results using MATLAB
**(1)** Run ‘Visualize_SPTnet_Outputs.m’

**(2)** Select the files used for testing the model

**(3)** Select the SPTnet inference results.

**(4)** By default, the tested videos with ground truth trajectories, Hurst exponent, and generalized diffusion coefficient will be shown in red, and the SPTnet estimation results will show different colors for different tracks. An example frame from the visualization result is showing below.

![image](https://github.com/user-attachments/assets/76d0af8e-cc4e-4d85-b89c-32b7d4b9bf22)


