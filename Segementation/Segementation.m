% Script for segement large video into smaller blocks
%
% (C) Copyright 2025                The Huang Lab
%
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
%
%     Author: Cheng Bi, July 2025

%% User Parameters
inputVarName    = 'timelapsedata';   % Name of the video variable
blockSizeXY     = [64, 64];         % [height, width] of segementaed spatial blocks
blockSizeT      = 30;               % number of frames per temporal block
output_folder_name = 'segementation_results';         % directory to save patches
baseName        = 'block';            % base filename for patches
paddingMethod   = 'zero';            % 'zero' or 'replicate'

%% Prepare data and output directory
if ~exist(output_folder_name, 'dir')
    mkdir(output_folder_name);
end

% Retrieve the data variable
data = evalin('base', inputVarName);
[H, W, T, N] = size(data);

%% Compute padding sizes to make dimensions divisible by block sizes
remH = mod(H, blockSizeXY(1));
remW = mod(W, blockSizeXY(2));
remT = mod(T, blockSizeT);

padH = (blockSizeXY(1) - remH) * (remH>0);
padW = (blockSizeXY(2) - remW) * (remW>0);
padT = (blockSizeT      - remT) * (remT>0);

% Choose padding value or method
switch paddingMethod
    case 'zero'
        padVal = 0;
    case 'replicate'
        padVal = 'replicate';
    otherwise
        error('Unknown paddingMethod: %s', paddingMethod);
end

% Apply padding (only along first three dimensions)
paddedData = padarray(data, [padH, padW, padT, 0], padVal, 'post');

% New dimensions after padding\ n
[Hp, Wp, Tp, ~] = size(paddedData);
numBlocksX = Hp / blockSizeXY(1);
numBlocksY = Wp / blockSizeXY(2);
numBlocksT = Tp / blockSizeT;

fprintf('Original size: %dx%dx%dx%d\n', H, W, T, N);
fprintf('Padded   size: %dx%dx%dx%d\n', Hp, Wp, Tp, N);
fprintf('Splitting into %dx%dx%d blocks per volume.\n', numBlocksX, numBlocksY, numBlocksT);

%% Perform segmentation and save patches
for n = 1:N
    vol = paddedData(:,:,:,n);
    for ix = 1:numBlocksX
        xRange = (ix-1)*blockSizeXY(1) + (1:blockSizeXY(1));
        for iy = 1:numBlocksY
            yRange = (iy-1)*blockSizeXY(2) + (1:blockSizeXY(2));
            for it = 1:numBlocksT
                tRange = (it-1)*blockSizeT + (1:blockSizeT);

                % Extract patch
                timelapsedata = vol(xRange, yRange, tRange);

                % Filename: e.g. data001_x1_y2_t2.mat
                fname = sprintf('%s%03d_x%d_y%d_t%d.mat', baseName, n, ix, iy, it);
                save(fullfile(output_folder_name, fname), 'timelapsedata');
            end
        end
    end
end

fprintf('Done: %d data files → %d patches saved in "%s".\n', N, N * numBlocksX * numBlocksY * numBlocksT, output_folder_name);

%%
