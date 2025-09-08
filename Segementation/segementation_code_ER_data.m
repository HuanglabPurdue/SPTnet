%load_and_crop_SLBdata
%% load raw tracking data
clear
clc
startingFolder = 'K:\SPT_2023\Experimental data\20241030_Chengspt sec61\';
defaultFileName = fullfile(startingFolder);
[baseFileName folder] = uigetfile([defaultFileName,'.mat'],'MultiSelect','on', 'Select mat files');
if size(baseFileName,2) == 0
  % User clicked the Cancel button.
  return;
end
filename = fullfile(folder, baseFileName);
load(filename)
rawdata = ims;

% convert the data to photon
% EMgain_setting = 250;
% [conversion,offset] = KLS_gain_basic(EMgain_setting); 
% rawdata_converted = (rawdata-offset)*conversion; % conversion is 1/gain

%% crop a 256 by 256

dipshow(max(rawdata,[],3))
diptruesize(200)
hold on 

h_rect = imrect(gca, [7 7 255 255]);
pause
% Rectangle position is given as [xmin, ymin, width, height]
pos_rect = h_rect.getPosition();
close(gcf)
% Round off so the coordinates can be used as indices
pos_rect_round = round(pos_rect);
% Select part of the image
dipshow(rawdata)
diptruesize(200)
hold on

rectangle('Position',pos_rect_round,'EdgeColor','r')
% text(pos_rect_round+5,pos_rect_round+5,num2str(1),'Color','red','FontSize',14)


% plot for the entire recorded period
subregion = rawdata(pos_rect_round(2) + (0:pos_rect_round(4)), pos_rect_round(1) + (0:pos_rect_round(3)),:);
dipshow(subregion,'lin')
diptruesize(200)

%% crop the subregion corresponding SR image
maxp_subregion = max(subregion,[],3);
startingFolder = 'K:\SPT_2023\Experimental data\20241030_Chengspt sec61\sec61_FOV2\smlm04\';
defaultFileName = fullfile(startingFolder);
[baseFileName folder] = uigetfile([defaultFileName,'.tif'],'MultiSelect','off', 'Select SR image');
if size(baseFileName,2) == 0
  % User clicked the Cancel button.
  return;
end
filename = fullfile(folder, baseFileName);
SRimage = imread(filename);
downsampled_SRimage = imresize(SRimage, 0.5);
zoom_in = 20; % Parameter needs to change based on the reconstruction settings!!!
maxp_subregion_interp = single(imresize(maxp_subregion, [5120,5120], 'bilinear'));
crossCorrelation = normxcorr2(maxp_subregion_interp, SRimage);
% Find the peak in cross-correlation
[maxCorr, maxIndex] = max(abs(crossCorrelation(:)));
[ypeak, xpeak] = ind2sub(size(crossCorrelation), maxIndex(1));

% Calculate the offset
yoffSet = ypeak - size(maxp_subregion_interp, 1);
xoffSet = xpeak - size(maxp_subregion_interp, 2);
% Define the crop rectangle
cropRect = [xoffSet + 1, yoffSet + 1, size(maxp_subregion_interp, 2) - 1, size(maxp_subregion_interp, 1) - 1];

% Crop the region
croppedImage = imcrop(SRimage, cropRect);
SRcolor_map = imfinfo(filename).Colormap;
% Display the cropped image
figure;
% flippedSRImage = flip(croppedImage, 1);
imshow(croppedImage,SRcolor_map);
title('Cropped Image from Large Image');

%% save to multiple files
ims_all = [];
for i = 1:7
    for j = 1:7
        for t = 1:33
            ims_all(i,j,t,:,:,:) = subregion((1+(i-1)*32):32+i*32,(1+(j-1)*32):32+j*32,(1+(t-1)*30):t*30);
        end
    end
end

ims = [];
position = pos_rect_round(:);
for xx = 1:7
    for yy = 1:7 
        ims = squeeze(ims_all(xx,yy,:,:,:,:));
        savedfilename = [folder,baseFileName,num2str(xx),'_',num2str(yy),'_cropped_256_64chunks_example1.mat'];
        save(savedfilename,'ims','position','croppedImage','-v7.3')
    end
end
