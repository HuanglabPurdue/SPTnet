% Script for segement large video into smaller blocks for SPTnet to process
%
% (C) Copyright 2025                The Huang Lab
%
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
%
%     Author: Cheng Bi, July 2025
%     Update: overlap support, Sept 2025

%% User Parameters
inputVarName        = 'timelapsedata';   % Name of the video variable (4D: H x W x T x N) (Height, Width, Time, Number of video)
blockSizeXY         = [64, 64];          % [height, width] of segementaed spatial blocks
blockSizeT          = 30;                % number of frames per temporal block
overlapXY           = [0, 0];            % spatial overlap in pixels [oy_rows, ox_cols]; 0 = no overlap
overlapT            = 0;                 % temporal overlap in frames; 0 = no overlap
output_folder_name  = 'segementation_results';  % directory to save patches
baseName            = 'block';           % base filename for patches
paddingMethod       = 'zero';            % 'zero' or 'replicate'

%% Load the video to be segmented
[file, path] = uigetfile('*.mat', 'Select the test data file');
fullFilePath = fullfile(path, file)
load(fullFilePath)
%% Prepare data and output directory
if ~exist(output_folder_name, 'dir')
    mkdir(output_folder_name);
end

% Retrieve the data variable
data = evalin('base', inputVarName);
[H, W, T, N] = size(data);

%% Resolve stride from overlaps
if any(overlapXY < 0) || overlapT < 0
    error('overlapXY and overlapT must be >= 0.');
end
strideXY = blockSizeXY - overlapXY;   % [row_stride, col_stride]
strideT  = blockSizeT  - overlapT;

if any(strideXY <= 0) || strideT <= 0
    error(['Overlap too large: need to smaller than blocksize. ', ...
           'Got strideXY=%s, strideT=%d. Reduce overlap or increase block size.'], ...
           mat2str(strideXY), strideT);
end

%% Compute how many blocks are needed and how much padding to cover the grid
numBlocksAlong = @(Len, blk, str) max(1, ceil((max(Len, blk) - blk) / str) + 1);

numBlocksX = numBlocksAlong(H, blockSizeXY(1), strideXY(1));  % along rows (height)
numBlocksY = numBlocksAlong(W, blockSizeXY(2), strideXY(2));  % along cols (width)
numBlocksT = numBlocksAlong(T, blockSizeT,      strideT);     % along time

reqH = (numBlocksX - 1) * strideXY(1) + blockSizeXY(1);
reqW = (numBlocksY - 1) * strideXY(2) + blockSizeXY(2);
reqT = (numBlocksT - 1) * strideT      + blockSizeT;

padH = max(0, reqH - H);
padW = max(0, reqW - W);
padT = max(0, reqT - T);

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

% New dimensions after padding
[Hp, Wp, Tp, ~] = size(paddedData);

fprintf('Original size: %dx%dx%dx%d\n', H, W, T, N);
fprintf('Padded   size: %dx%dx%dx%d\n', Hp, Wp, Tp, N);
fprintf('Block size   : [%d %d] x %d (T)\n', blockSizeXY(1), blockSizeXY(2), blockSizeT);
fprintf('Overlap      : [%d %d] x %d (T)\n', overlapXY(1), overlapXY(2), overlapT);
fprintf('Stride       : [%d %d] x %d (T)\n', strideXY(1), strideXY(2), strideT);
fprintf('Splitting into %dx%dx%d blocks per volume.\n', numBlocksX, numBlocksY, numBlocksT);

%% Perform segmentation and save patches
for n = 1:N
    vol = paddedData(:,:,:,n);
    for ix = 1:numBlocksX
        xRange = ( (ix-1)*strideXY(1) + 1 ) : ( (ix-1)*strideXY(1) + blockSizeXY(1) );
        for iy = 1:numBlocksY
            yRange = ( (iy-1)*strideXY(2) + 1 ) : ( (iy-1)*strideXY(2) + blockSizeXY(2) );
            for it = 1:numBlocksT
                tRange = ( (it-1)*strideT + 1 ) : ( (it-1)*strideT + blockSizeT );

                % Extract patch
                timelapsedata = vol(xRange, yRange, tRange);

                % Filename: e.g. block001_x1_y2_t2.mat
                fname = sprintf('%s%03d_x%d_y%d_t%d.mat', baseName, n, ix, iy, it);
                save(fullfile(output_folder_name, fname), 'timelapsedata');
            end
        end
    end
end

% Save a small settings struct for reproducibility (file name per your preference)
segementation_settings = struct();
segementation_settings.inputVarName  = inputVarName;
segementation_settings.blockSizeXY   = blockSizeXY;
segementation_settings.blockSizeT    = blockSizeT;
segementation_settings.overlapXY     = overlapXY;
segementation_settings.overlapT      = overlapT;
segementation_settings.strideXY      = strideXY;
segementation_settings.strideT       = strideT;
segementation_settings.originalSize  = [H W T N];
segementation_settings.paddedSize    = [Hp Wp Tp N];
segementation_settings.numBlocksXYZT = [numBlocksX numBlocksY numBlocksT N];
save(fullfile(output_folder_name, 'segementation_settings.mat'), 'segementation_settings');

fprintf('Done: %d data files → %d patches saved in "%s".\n', ...
    N, N * numBlocksX * numBlocksY * numBlocksT, output_folder_name);
