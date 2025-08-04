# SPTnet (single-particle tracking neural network)
This repository accompanies the manuscript:
“SPTnet: a deep learning framework for end-to-end single-particle tracking and motion dynamics analysis”
by Cheng Bi, Kevin L. Scrudders, Yue Zheng, Maryam Mahmoodi, Shalini T. Low-Nam, and Fang Huang.


## Repository Structure
```text
├── Python files
│   ├── SPTnet_toolbox.py          # SPTnet architecuture and Utilities for data loading
│   ├── SPTnet_training.py         # Training script for SPTnet
│   ├── SPTnet_inference.py        # Inference script for trained model
│   ├── transformer.py             # Spatial transformer module (Vaswani, A., et al., Attention is all you need. Advances in neural information processing systems, 2017)
│   ├── transformer3D.py           # Temporal transformer module (Modified to take 3D inputs)
│   └── mat_to_tiff.py             # Converts .mat videos to TIFF series
│
├── MATLAB files
│   ├── SPTnet_trainingdata_generator.m        # GUI to generate training datasets
│   ├── Visualize_SPTnet_Outputs_GUI.m         # GUI to visualize inference results
│   └── CRLB_H_D_frame.mat                     # CRLB matrix used in loss function
│
├── Example_data
│   ├── Example_test_data/        # mat file contains 10 test videos and an example TIFF series
│   └── Example_trainingdata/     # mat file contains 100 simulated videos for training
│
├── Trained_models               # Pretrained model (based on a Nikon Ti2 TIRF system, NA=1.49)
├── PSF-toolbox                  # Simulated PSF generation toolbox
├── DIPimage_2.9                 # DIPimage toolbox for MATLAB (https://diplib.org/download_2.9.html)
├── Segmentation                 # Scripts for ER protein segmentation and stitching
├── SPTnet_environment.yml       # Conda environment configuration included all necessary packages for SPTnet
├── package_required_for_SPTnet.txt   # packages required by SPTnet
├── Installation of SPTnet.pdf     
└── SPTnet user manual.pdf
 ```

## 2. Instructions for generating training videos
This code has been tested on the following systems and packages:
Microsoft Windows 10 Education, Matlab R2021a, DIPimage 2.8.1 (http://www.diplib.org/)

**(1)** Change MATLAB current folder to the directory that contains “PSF-toolbox”.
Run ‘SPTnet_trainingdata_generator.m’ to generate the training dataset.

![image](https://github.com/user-attachments/assets/0d91fba5-65ad-4795-b75d-bcf09ece50b6)

**(2)** The default settings will generate 5 files each containing 100 videos.

![image](https://github.com/user-attachments/assets/51f965b0-6846-447f-b568-bc67e5745a35)

**Note:** SPTnet was trained with >200,000 videos to achieve precision approching CRLB. To generate more training and validation datasets, please locate the variables specifying the number of training videos in ‘SPTnet_trainingdata_generator.m’ (e.g., total_files, numvideos). Increase these values to the desired amount, then run the script again. This will produce additional .mat files containing more simulated videos for training and validation.

## 3. Instructions for training SPTnet using simulated training datasets
The code has been tested on the following systems and packages:
Ubuntu20.04LTS, Python3.9.12, Pytorch1.11.0, CUDA11.3, MatlabR2021a

**To start training,**

**(1)** Type the following command in the terminal: python SPTnet_training.py

**(2)** Select the folder to save the trained model

**(3)** Select the training data files.

**(4)** During the training, the model with the minimal validation loss will be saved as ‘trained_model’ onto the selected folder in step (2), together with an image of the training loss and validation loss changes along with training epoch.

![model1_04NAlearning curve2](https://github.com/user-attachments/assets/59f4dad5-2fdf-44c6-852e-f56112a592a3)


## 4. Instructions for running inference using a trained model
To test the trained model,

**(1)** Type the following command in terminal: python SPTnet_inference.py

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


