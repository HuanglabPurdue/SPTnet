# SPTnet (single-particle tracking neural network)
This repository accompanies the manuscript:
“SPTnet: a deep learning framework for end-to-end single-particle tracking and motion dynamics analysis”
by Cheng Bi, Kevin L. Scrudders, Yue Zheng, Maryam Mahmoodi, Shalini T. Low-Nam, and Fang Huang.


## Files included in this package
### Content of SPTnet software (Matlab scripts/file)
* SPTnet_trainingdata_generator.m: Matlab code to generate simulated videos for SPTnet training
* Visualize_SPTnet_Outputs.m: Matlab code to visualize the final output from SPTnet.
* CRLB_H_D_frame.mat: Calculated CRLB matrix of Hurst exponent and generalized diffusion coefficient used in the loss function

### Content of SPTnet software (Python)
* SPTnet_toolbox.py: Script for SPTnet architecture and other tools used in loading data and output result
* SPTnet_training.py: Script to train the SPTnet
* SPTnet_infernece.py: Script to use a trained model for inference
* transformer.py: The same transformer module used in DETR (Caron,N.,et al 2020) (spatial-T)
* transformer3D.py: The transformer module modified to take 3D inputs (temporal-T)

### Others
Example test data: Containing 10 test videos in one file
PSF-toolbox: Toolbox used to simulate PSF through pupil function
Trained models: Containing a pre-trained model based on the parameters of a Nikon Ti2 TIRF microscope

## 1. Instructions for generating training videos
This code has been tested on the following systems and packages:
Microsoft Windows 10 Education, Matlab R2021a, DIPimage 2.8.1 (http://www.diplib.org/)
Change MATLAB current folder to the directory that contains “PSF-toolbox”.
Run ‘SPTnet_trainingdata_generator.m’ to generate the training dataset.
![image](https://github.com/user-attachments/assets/0d91fba5-65ad-4795-b75d-bcf09ece50b6)

The default settings will generate 5 files each containing 100 videos.

## 2. Instructions for training SPTnet using simulated training datasets
The code has been tested on the following systems and packages:
Ubuntu20.04LTS, Python3.9.12, Pytorch1.11.0, CUDA11.3, MatlabR2021a
To start training,
(1) type the following command in the terminal: python SPTnet_training.py
(2) Select the folder to save the trained model
(3) Select the training data files.
(4) During the training, the model with the minimal validation loss will be saved as ‘trained_model’ onto the selected folder in step (2), together with an image of the training loss and validation loss changes along with training epoch.

## 3. Instructions for running inference using a trained model
To test the trained model,
(1) Type the following command in terminal: python SPTnet_inference.py
(2) Select the trained model that will be used to conduct the inference
(3) Select the video file that will be analyzed by SPTnet
(4) An output ‘.mat’ file will be generated under the ‘inference_results’ folder located in the directory of the selected model in step (2), which contains all the estimated trajectories, detection probabilities, Hurst exponents, and generalized diffusion coefficients ready for analysis or visualization.

## 4. Instructions for visualizing the SPTnet outputs overlay with ground truth in a videos
(1) Run ‘Visualize_SPTnet_Outputs.m’
(2) Select the files used for testing the model
(3) Select the SPTnet inference results.
(4) By default, the tested videos with ground truth trajectories, Hurst exponent, and generalized diffusion coefficient will be shown in red, and the SPTnet estimation results will show different colors for different tracks. An example frame from the visualization result is attached below.
![image](https://github.com/user-attachments/assets/76d0af8e-cc4e-4d85-b89c-32b7d4b9bf22)

