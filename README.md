# SPTnet (single-particle tracking neural network)
This repository accompanies the manuscript:
“SPTnet: a deep learning framework for end-to-end single-particle tracking and motion dynamics analysis”
by Cheng Bi, Kevin L. Scrudders, Yue Zheng, Maryam Mahmoodi, Shalini T. Low-Nam, and Fang Huang.


## Files included in this package
### Content of SPTnet software (Matlab scripts/file)
* SPTnet_trainingdata_generator.m :Matlab code to generate simulated videos for SPTnet training
* Visualize_SPTnet_Outputs.m :Matlab code to visualize the final output from SPTnet.
* CRLB_H_D_frame.mat :Calculated CRLB matrix of Hurst exponent and generalized diffusion coefficient used in the loss function

### Content of SPTnet software (Python)
* SPTnet_toolbox.py :Script for SPTnet architecture and other tools used in loading data and output result
* SPTnet_training.py :Script to train the SPTnet
* SPTnet_infernece.py :Script to use a trained model for inference
* transformer.py :The transformer module used as spatial-T
* transformer3D.py :The transformer module used as temporal-T
