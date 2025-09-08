clear
clc

% load raw input
startingFolder = 'K:\SPT_2023\Experimental data\20241030_Chengspt sec61\';
defaultFileName = fullfile(startingFolder);
[baseFileName folder] = uigetfile([defaultFileName,'.mat'],'MultiSelect','on', 'Select Raw input files');
if size(baseFileName,2) == 0
  % User clicked the Cancel button.
  return;
end
datanum = size(baseFileName,2);
for i = 1:datanum
    filename = fullfile(folder, baseFileName{i});
    xx = mod(i-1, 7) + 1;
    yy = floor((i-1) / 7) + 1;
    ims_all(yy,xx,:,:,:,:) = load(filename).ims;
end
pos_rect = load(filename).position;
% % SRpos_rect = load(filename).SRpos_rect;
% movingTransformed = load(filename).movingTransformed;
% load('K:\SPT_2023\Experimental data\20240411_antiCD3-AF647_Diffusion_Jurkat_01\01_SA-AF647_200pM_Diffusion_nice\03_Diffusion_SA-AF647_Stream_21msInterval_10gain_05mW_20sdur.mat_cropped1.mat')
% % load SPTNet estimation
startingFolder = 'K:\SPT_2023\Multi-objects tracking\Attention_SPTnet\ourTIRF_model\model2_ourTIRF_linux2_unfinished0.123\';
defaultFileName = fullfile(startingFolder);
[baseFileName folder] = uigetfile([defaultFileName,'.mat'],'MultiSelect','on', 'Select SPTNet estimation files');
if size(baseFileName,2) == 0
  % User clicked the Cancel button.
  return;
end
for i = 1:datanum
    filename = fullfile(folder, baseFileName{i});
    load(filename);
    xx = mod(i-1, 7) + 1;
    yy = floor((i-1) / 7) + 1;
    estimation_xy_all(yy,xx,:,:,:,:) = estimation_xy;
    estimation_C_all(yy,xx,:,:) = estimation_C;
    estimation_H_all(yy,xx,:,:) = estimation_H;
    obj_estimation_all(yy,xx,:,:,:,:) = obj_estimation;
end
%
zoom_in = 20;% zoom in 
estimation_xy_scale_all = (estimation_xy_all*32+32);%*zoom_in;
estimation_C_all = estimation_C_all; 
% data format transfer for linux pc trained model
estimation_xy_perm_all = permute(estimation_xy_scale_all,[1,2,3,5,4,6]);
obj_estimation_all = permute(obj_estimation_all,[1,2,3,6,5,4]);

%% display raw estimation
num_queries = 20;
cmap = turbo(1000);
frmlist = 1:30;
% dipshow(movingTransformed,'lin')
% diptruesize(75)

% minpixel = min(ims(:,:,:),[],'all');
% maxpixel = max(ims(:,:,:),[],'all');
% frm=uint8(((ims(:,:,frame,data_num)-minpixel)/(maxpixel-minpixel))*255);
% rgb_img = gray2rgb(frm);
% imshow(rgb_img,'InitialMagnification',2000)

hold on
for xx = 1:7
    for yy = 1:7
        for data_num =1:33
            threshold = 0.95;
            frame_num = 30;
            cropsize =5;
            d = cropsize/2;
            predict = squeeze(obj_estimation_all(xx,yy,data_num,:,:)>threshold);
            C_all = [];
            H_all = [];
            for i = 1:num_queries
                if sum(obj_estimation_all(xx,yy,data_num,:,i)>=threshold)>=10
                   Hidx = int16(round(estimation_H_all(xx,yy,data_num,i),3)*1000);
                   Cidx = int16(round(estimation_C_all(xx,yy,data_num,i),3)*1000);
                   plot(squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,1))+1+32*yy,squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,2))+1+32*xx,'-o','MarkerSize',2,'color',cmap(Hidx,:),'LineWidth',1,'MarkerFaceColor',cmap(Hidx,:))
                   hold on
        %             if estimation_H(data_num,i)~=0
        %                 count = count+1;
        %                 C_all(count) = round(estimation_C(data_num,i),2);
        %                 H_all(count) = round(estimation_H(data_num,i),2);
                end
            end
        end
    end
end
overlay_result = getframe(gcf);
% imwrite(overlay_result.cdata, ['K:\SPT_2023\Multi-objects tracking\Attention_SPTnet\model1_ourTIRF_linux4\experimentaldata_result_ER_allfrm\', 'FOV2_allfrm_10_H.tiff']);
% dipshow(overlay_frame.cdata)
% savedfilename = [folder,'/',num2str(datafilename),'.mat'];
% save(savedfilename,'H_all','C_all','-v7.3')

% imagesc(timelapsedata(:,:,data_num))

%% Reject wrong trajectories
keeped_H = {};
keeped_C = {};
keeped_tracks = {};
keeped_Hidx = {};
keeped_Cidx = {};
wrongtrack = 0;
total_track = 0;
for data_num =1:33
    count = 1;
    for xx = 1:7
        for yy = 1:7
            threshold = 0.95;
            frame_num = 30;
            cropsize =5;
            d = cropsize/2;
            predict = squeeze(obj_estimation_all(xx,yy,data_num,:,:)>threshold);
            C_all = [];
            H_all = [];
            for i = 1:num_queries
                if sum(obj_estimation_all(xx,yy,data_num,:,i)>=threshold)>=10 % only demonstrate tracks last more than x frames
                   trace = [squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,1))+1+32*yy,squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,2))+1+32*xx];
                   dx = diff(trace(:,1));
                   dy = diff(trace(:,2));
                   distances = sqrt(dx.^2 + dy.^2);
                   if any((distances>=5)==1)==0 % reject tracks with wrong linkage larger than N pixels
                       keeped_H{data_num,count} = estimation_H_all(xx,yy,data_num,i);
                       keeped_C{data_num,count} = estimation_C_all(xx,yy,data_num,i);
                       keeped_tracks{data_num,count} = trace;
                       Hidx = int16(round(estimation_H_all(xx,yy,data_num,i),3)*1000);
                       Cidx = int16(round(estimation_C_all(xx,yy,data_num,i),3)*1000);
                       keeped_Hidx{data_num,count} = Hidx;
                       keeped_Cidx{data_num,count} = Cidx;
                       plot(squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,1))+1+32*yy,squeeze(estimation_xy_perm_all(xx,yy,data_num,[frmlist(predict(:,i))],i,2))+1+32*xx,'-o','MarkerSize',2,'color',cmap(Cidx,:),'LineWidth',1,'MarkerFaceColor',cmap(Cidx,:))
                       hold on
            %             if estimation_H(data_num,i)~=0
            %                 count = count+1;
            %                 C_all(count) = round(estimation_C(data_num,i),2);
            %                 H_all(count) = round(estimation_H(data_num,i),2);
                       count = count+1;
                       total_track = total_track+1;
                   else
                       wrongtrack = wrongtrack+1;
                   end
                end
            end
        end
    end
end

%% get rid of repeated trajectories
cellTracks = keeped_tracks; %  cell array with tracks data)
% Parameters
numOverlapSteps = 5;
distThreshold = 1;
pixel_size =0.108; % um
% Remove repeated tracks
[cellTracks, num_rep,keeped_H_f,keeped_C_f] = removeRepeatedTracks(cellTracks, numOverlapSteps, distThreshold,keeped_H,keeped_C);
cmap_frame = turbo(size(cellTracks,1));
% Display by frame chunks
for frame_chunks =1:size(cellTracks,1)
    for track_num = 1: size(cellTracks,2)
        if isempty(cellTracks{frame_chunks,track_num})==0 
            plot(cellTracks{frame_chunks,track_num}(:,1)*pixel_size,cellTracks{frame_chunks,track_num}(:,2)*pixel_size,'-o','color',cmap_frame(frame_chunks,:),'MarkerSize',1,'LineWidth',1)
            hold on
        end
    end
end
fig = gcf;
fig.Position = [400, 50, 1024, 1024];
axis tight
axis equal
title('Color coded by time');
xlabel('X-axis (\mum)');
ylabel('Y-axis (\mum)');

%% display by Hurst exponent
cmap_H = turbo(1000);
for frame_chunks =1:size(cellTracks,1)
    for track_num = 1:size(cellTracks,2)
        if isempty(cellTracks{frame_chunks,track_num})==0 %& keeped_Hidx{frame_chunks,track_num}>800
            plot(cellTracks{frame_chunks,track_num}(:,1)*pixel_size,cellTracks{frame_chunks,track_num}(:,2)*pixel_size,'-o','color',cmap_H(keeped_Hidx{frame_chunks,track_num},:),'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_H(keeped_Hidx{frame_chunks,track_num},:))
            hold on
        end
    end
end

fig = gcf;
fig.Position = [400, 50, 1024, 1024];
axis tight
axis equal
title('Color coded by Hurst exponent');
xlabel('X-axis (\mum)');
ylabel('Y-axis (\mum)');
set(gca, 'YDir', 'reverse');

%% display by Hurst exponent, Movie for different frame chunks
% Define the colormap
cmap_H = turbo(1000);
% Calculate the overall extent of the coordinates
min_x = Inf;
max_x = -Inf;
min_y = Inf;
max_y = -Inf;
% Determine the overall extent by iterating through all frames
for frame_chunks = 1:33
    for track_num = 1:size(cellTracks, 2)
        if ~isempty(cellTracks{frame_chunks, track_num})
            coords = cellTracks{frame_chunks, track_num};
            min_x = min(min_x, min(coords(:,1)));
            max_x = max(max_x, max(coords(:,1)));
            min_y = min(min_y, min(coords(:,2)));
            max_y = max(max_y, max(coords(:,2)));
        end
    end
end

% Define the figure size based on the maximum extent
margin = 15; % Add margin around the plot
figureWidth = 800;  % Width of the figure window
figureHeight = 800; % Height of the figure window

% Create a VideoWriter object to write the video file
videoFilename = 'trajectory_movie.avi'; % Specify the output filename
videoWriter = VideoWriter(videoFilename); % Create the VideoWriter object
videoWriter.FrameRate = 5; % Set the frame rate (frames per second)
open(videoWriter); % Open the video writer object

% Loop through each frame
for frame_chunks = 1:33
    % Create a new figure for each frame with a fixed size
    figure('Position', [100, 100, figureWidth, figureHeight]);
    % Loop through each track
    for track_num = 1:size(cellTracks, 2)
        if ~isempty(cellTracks{frame_chunks, track_num}) 
            % Plot the trajectory
            plot(cellTracks{frame_chunks, track_num}(:,1) * pixel_size, ...
                 cellTracks{frame_chunks, track_num}(:,2) * pixel_size, ...
                 '-o', 'color', cmap_H(keeped_Hidx{frame_chunks, track_num}, :), ...
                 'MarkerSize', 1, 'LineWidth', 1, 'MarkerFaceColor', cmap_H(keeped_Hidx{frame_chunks, track_num}, :));
            hold on;
        end
    end
    
    % Set the axis properties to fit the data
    xlim([min_x - margin, max_x + margin] * pixel_size);
    ylim([min_y - margin, max_y + margin] * pixel_size);
%     axis equal;
    title(sprintf('Frame %d - Color coded by Hurst exponent', frame_chunks));
    xlabel('X-axis (\mum)');
    ylabel('Y-axis (\mum)');
    set(gca, 'YDir', 'reverse');
    
    % Capture the current figure
    frame = getframe(gcf); % Capture the entire figure window
    writeVideo(videoWriter, frame); % Write the frame to the video file
    
    % Close the current figure
    close(gcf);
end

% Finalize the video file
close(videoWriter);
disp('Movie creation complete.');

%% display by diffusion coefficient
cmap_D = turbo(1000);
for frame_chunks =1:size(cellTracks,1)
    for track_num = 1: size(cellTracks,2)
        if isempty(cellTracks{frame_chunks,track_num})==0 %& keeped_C{frame_chunks,track_num}>0.4
            plot(cellTracks{frame_chunks,track_num}(:,1)*pixel_size,cellTracks{frame_chunks,track_num}(:,2)*pixel_size,'-o','color',cmap_D(keeped_Cidx{frame_chunks,track_num},:),'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_D(keeped_Cidx{frame_chunks,track_num},:))
            hold on
        end
    end
end

fig = gcf;
fig.Position = [400, 50, 1024, 1024];
axis tight
axis equal
title('Color coded by generalized diffusion coefficient');
xlabel('X-axis (\mum)');
ylabel('Y-axis (\mum)');
set(gca, 'YDir', 'reverse');

%% display selected subregion
% generate diffraction limited subregion
cmap_H = jet(1000);
addpath('E:\Git_cheng\Live3DSMSN\SRsCMOS')
addpath('E:\Git_cheng\helpers')
addpath('E:\Git_cheng\mex')
datapath = 'K:\SPT_2023\Experimental data\20241030_Chengspt sec61\sec61_FOV2';
load('K:\SPT_2023\Experimental data\20241030_Chengspt sec61\sec61_FOV2\smlm04\recon_final_backup.mat')
% load('K:\SPT_2023\Experimental data\20241030_Chengspt sec61\sec61_FOV4_4_10ms\smlm0405\recon_final_backup.mat')
% load('K:\SPT_2023\Experimental data\20241106 Cheng_spt sec61\FOV1\smlm04\recon_final_backup.mat')
srobj.resultfolder = [datapath];
srobj.imagesz = 512;
srobj.zm = 20;
srobj.Cam_pixelsz = 108;

srobj.GenerateDiffLimImage()
diff_image = double(srobj.difflim);
diff_image_subregion = diff_image(pos_rect(2)-5 + (0:pos_rect(4)), pos_rect(1)-4 + (0:pos_rect(3)),:);
adjustedImage = imadjust(uint8(diff_image_subregion));
% h = imshow(uint8(diff_image_subregion));
% adjustedImage = flipud(adjustedImage);
% imshow(adjustedImage)
darkerImage = imadjust(adjustedImage, [0 1], [0 0.8]); % Adjust intensity range
figure 
set(gcf, 'WindowState', 'maximized');
imshow(darkerImage,'InitialMagnification','fit');
% [y, x] = find(cleanedSkeleton);
% plot(x,y,'.','color','red')
% Set the y-axis direction to reverse
hold on
% set(h, 'AlphaData', 0.1);  % Adjust transparency if needed
%overlay tracks with diffraction limited images
% allNumbers = [];
% for i = 1:numel(cellTracks)
%     if isempty(cellTracks{i})==0
%         allNumbers_x = [allNumbers_x; cellTracks{i}(:,1)];
%         allNumbers_y = [allNumbers_y; cellTracks{i}(:,1)];
%     end
% end
% minx = min(allNumbers_x);
% miny = min(allNumbers_y);
cmap_D = turbo(1000);
cmap_H = turbo(1000);
for frame_chunks =1:size(cellTracks,1)
    for track_num = 1:size(cellTracks,2)
        if isempty(cellTracks{frame_chunks,track_num})==0  %& keeped_C{frame_chunks,track_num}<0.05 % | keeped_H{frame_chunks,track_num}<0.3
             plot(cellTracks{frame_chunks,track_num}(:,1)-32-0.1,cellTracks{frame_chunks,track_num}(:,2)-32+0.5,'-','color',cmap_H(keeped_Hidx{frame_chunks,track_num},:),'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_H(keeped_Hidx{frame_chunks,track_num},:))
%              plot(cellTracks{frame_chunks,track_num}(:,1)-32-0.1,cellTracks{frame_chunks,track_num}(:,2)-32+0.5,'-','color',[cmap_D(keeped_Cidx{frame_chunks,track_num},:),0.5],'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_D(keeped_Cidx{frame_chunks,track_num},:))
            hold on
        end
    end
end
% ax = gca;
% x_limits = ax.XLim;
% y_limits = ax.YLim;
% xlim([0 512])
% ylim([0 512])
% set(gcf, 'WindowState', 'maximized');

% for i = 1:num_roi
%     rectangle('Position',pos_rect_all(i,:),'EdgeColor','w', 'LineWidth',2,'LineStyle','--')
%     text(pos_rect_all(i,1)+2,pos_rect_all(i,2)+5,num2str(i),'Color','w','FontSize',18,'FontName', 'Arial','fontweight','bold')
% end
% fig = gcf;
% 

% fig.Position = [400, 50, 1024, 1024];

%% select multiple subregions to display 
num_roi = 4;
% for i = 1:num_roi
%     dipshow(adjustedImage)
%     diptruesize(200)
%     hold on 
%     if i ~= 1
%         for j = 1:i-1
%             rectangle('Position',pos_rect_all(j,:),'EdgeColor','r')
%             text(pos_rect_all(j,1)+5,pos_rect_all(j,2)+5,num2str(j),'Color','red','FontSize',14)
%         end
%     end
%     h_rect = imrect(gca, [7 7 47 47]);
%     pause
%     % Rectangle position is given as [xmin, ymin, width, height]
%     pos_rect = h_rect.getPosition();
%     close(gcf)
%     % Round off so the coordinates can be used as indices
%     pos_rect_all(i,:) = round(pos_rect);
%     % Select part of the image
% end
dipshow(adjustedImage)
diptruesize(200)
hold on
cmap_subregion = jet(4);
% for i = 1:num_roi
%     rectangle('Position',pos_rect_all(i,:),'LineStyle','--','EdgeColor','w')
%     text(pos_rect_all(i,1)+2,pos_rect_all(i,2)+5,num2str(i),'Color','w','FontSize',14,'FontName', 'Arial','fontweight','bold')
% end
cmap_C = turbo(1000);
% display
% for subreg = 1:4%size(pos_rect_all,1)
%     figure
%     imshow(adjustedImage)
%     % Set the y-axis direction to reverse
%     hold on
%     for frame_chunks =1:size(cellTracks,1)
%         for track_num = 1:size(cellTracks,2)
%             if isempty(cellTracks{frame_chunks,track_num})==0
% %                 if keeped_Hidx{frame_chunks,track_num} > 700;
%                      plot(cellTracks{frame_chunks,track_num}(:,1)-32,cellTracks{frame_chunks,track_num}(:,2)-32,'-','color',[cmap_H(keeped_Hidx{frame_chunks,track_num},:),0.5],'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_H(keeped_Hidx{frame_chunks,track_num},:))
%     %                 plot(cellTracks{frame_chunks,track_num}(:,1)-32,cellTracks{frame_chunks,track_num}(:,2)-32,'-','color',[cmap_C(keeped_Cidx{frame_chunks,track_num},:),0.5],'MarkerSize',1,'LineWidth',1,'MarkerFaceColor',cmap_C(keeped_Cidx{frame_chunks,track_num},:))
%                     hold on
% %                 end
%             end
%         end
%     end
%     xlim([pos_rect_all(subreg,1), pos_rect_all(subreg,1) + pos_rect_all(subreg,3)]);
%     ylim([pos_rect_all(subreg,2), pos_rect_all(subreg,2) + pos_rect_all(subreg,4)]);
%     fig = gcf
%     fig.Position = [400, 50, 512, 512];
% end
%%
keeped_H_vector = [];
keeped_C_vector = [];
keeped_track_vector = {};
% Loop through each cell and concatenate the values
for i = 1:numel(keeped_H_f)
    keeped_H_vector = [keeped_H_vector, keeped_H_f{i}];
    keeped_C_vector = [keeped_C_vector, keeped_C_f{i}];
    keeped_track_vector = [keeped_track_vector,cellTracks{i}];
end
frameinterval = 0.01;
pixelsize = 0.108;
keeped_C_vector_um = (1/frameinterval).^(2*keeped_H_vector)*(pixelsize^2).*keeped_C_vector;
keeped_C_vector_um(keeped_H_vector<0.6 & keeped_H_vector>0.4)
histogram(keeped_C_vector_um(keeped_H_vector>0.4 &keeped_H_vector<0.6),15,'Normalization', 'probability')

histogram(keeped_H_vector,15,'Normalization', 'probability')
% xlim([0 5])
%% save
keeped_H_vector_sec61_10ms = keeped_H_vector
keeped_C_vector_sec61_10ms = keeped_C_vector
save('sec61_HD_10ms.mat', 'keeped_H_vector_sec61_10ms', 'keeped_C_vector_sec61_10ms')
%% fit 3 gaussian 
% Fit the data with a mixture of 3 Gaussians
rng(8)
%rng(0)
numGaussians = 1;
options = statset('MaxIter',5000, 'Display', 'final','TolFun',1e-4,'TolX',1.e-10);
gm = fitgmdist(keeped_H_vector', numGaussians, 'Options', options, 'Replicates', 100);
% gm = fitgmdist(keeped_H_vector', numGaussians);

% Display the parameters of the fitted Gaussian mixture model
disp(gm);

% Plot the histogram of the data and the fitted Gaussian mixture model
% figure('Position', [400 200 512 368]);
figure('Position', [400 200 1024 768]);
edges = [0:0.1:1]
histogram(keeped_H_vector, edges, 'Normalization', 'pdf','FaceColor', 'k', 'FaceAlpha', 0.15); % Plot histogram of data
hold on;

% Plot the fitted Gaussian mixture model
x = linspace(min(keeped_H_vector), max(keeped_H_vector), 1000);
y = pdf(gm, x');
plot(x, y, 'k-', 'LineWidth', 2, 'DisplayName', 'Gaussian Mixture Model');
idx = cluster(gm,keeped_H_vector');
% The value in idx(i) is the cluster index of observation i and indicates the component 
% with the largest posterior probability given the observation i.
population1 = keeped_C_vector(idx==1);
population2 = keeped_C_vector(idx==2);
population3 = keeped_C_vector(idx==3);

population1_H = keeped_H_vector(idx==1);
population2_H = keeped_H_vector(idx==2);
population3_H = keeped_H_vector(idx==3);
% Plot each Gaussian component separately
mixgaussiancmap = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
for k = 1:numGaussians
    mu = gm.mu(k);
    sigma = sqrt(gm.Sigma(k));
    p = gm.ComponentProportion(k);
    y_k = p * normpdf(x, mu, sigma);
    plot(x, y_k, 'LineWidth', 2,'color',mixgaussiancmap(k,:));
end

legend('Hurst by SPTNet', 'GMM fitting','Super-diffusion','Sub-diffusion','Brownian motion');
% title('Hurst exponent distribution - RTN4');
xlabel('Hurst exponent');
ylabel('Probability Density');
ax =gca
ax.LineWidth = 2
legend boxoff
legend('location','northwest')
set(gca,'FontSize',24,'FontName', 'Arial')

%%
% Parameters
numBootstraps = 200; % Number of bootstrap samples
numGaussians = 3; % Number of Gaussian components
proportions_bootstrap = zeros(numBootstraps, numGaussians); % To store proportions from each bootstrap
proportions_mu = zeros(numBootstraps, numGaussians);
% Original data
data = keeped_H_vector';

% Bootstrapping
for i = 1:numBootstraps
    % Resample data with replacement
    bootstrap_sample = datasample(data, size(data, 1));
    
    % Fit GMM to bootstrap sample
    gm_bootstrap = fitgmdist(bootstrap_sample, numGaussians, 'Options', options, 'Replicates', 200);
    
    % Store the mixture proportions
    proportions_bootstrap(i, :) = gm_bootstrap.ComponentProportion;
    proportions_mu(i,:) = gm_bootstrap.mu
end

% Calculate the mean and standard deviation of the proportions
mean_proportions = mean(proportions_bootstrap, 1);
std_proportions = std(proportions_bootstrap, 0, 1);

% Display the results
disp('Mean Proportions:');
disp(mean_proportions);
disp('Standard Deviation of Proportions:');
disp(std_proportions);
%% Sort proportions
for i = 1:size(proportions_mu, 1)
    % Get the indices that sort the i-th row of A in descending order
    [~, sort_idx] = sort(proportions_mu(i, :), 'ascend');
    
    % Use these indices to rearrange the i-th row of B
    proportions_bootstrap_sorted(i, :) = proportions_bootstrap(i, sort_idx);
end
mean_proportions = mean(proportions_bootstrap_sorted, 1);
std_proportions = std(proportions_bootstrap_sorted, 0, 1);
%% plot MSD for each populations

for i = 1:size(keeped_track_vector,2)
    track = keeped_track_vector{i}.*0.108; % Get the i-th track, which is a Nx2 matrix
    num_points = size(track, 1); % Number of points in the track
    
    % Initialize MSD for this track
    msd = zeros(num_points - 1, 1); 
    
    % Calculate MSD
    for tau = 1:(num_points - 1)
        displacements = track(1+tau:end, :) - track(1:end-tau, :);
        squared_displacements = displacements.^2;
        msd(tau) = mean(sum(squared_displacements, 2));
    end
    
    msd_all_tracks{i} = msd; % Store the MSD for this track
end
figure('Position', [400 200 600 768]);
hold on;
% msd_cmap = 
for pop = 1:3
    msd_pop = [];
    % Calculate the average MSD for this population
    msd_pop = msd_all_tracks(idx==pop);
    tau_all = nan(30,size(msd_pop,2));
    for ii= 1:size(msd_pop,2)
        for tau = 1:size(msd_pop{ii},1)
            tau_all(tau,ii) = msd_pop{ii}(tau);
        end
    end
    mean_tau_all = mean(tau_all,2,'omitnan')
    std_dev = std(tau_all, 0, 2,'omitnan');
    sem = std_dev / sqrt(sum(~isnan(tau_all),2));
%     errorplot(1:1:30,mean_tau_all)
%     plot(mean_tau_all(1:12))
%     figure;
    errorbar(1:10, mean_tau_all(1:10), sem(1:10), 'o-', 'LineWidth', 2, 'MarkerSize', 6);
    hold on
    % Plot the average MSD for this population    
end
xl=xlim;
yl=ylim;
line([xl(1),xl(2)],[yl(2),yl(2)],'color',[0 0 0],'LineWidth',2);   %画上边框，线条的颜色设置为黑色
line([xl(2),xl(2)],[yl(1),yl(2)],'color',[0 0 0],'LineWidth',2);
xticks([0 5 10 15 20])
xticklabels({'0','0.1','0.2','0.3','0.4'})
xlabel('Time (s)');
ylabel('MSD (um^2)');
legend('Super-diffusion','Sub-diffusion','Brownian motion')
legend box off
legend('location','northwest')
set(gca,'FontSize',20,'FontName', 'Arial')
fig =gcf
ax =gca
ax.LineWidth = 2
%% Bar plot for different population
figure('Position', [400 200 600 768]);
% x = ["Sub-diffusion" "Brownian motion" "Super-diffusion"];
x = categorical({'Sub-diffusion','Brownian motion','Super-diffusion'});
x = reordercats(x,{'Sub-diffusion','Brownian motion','Super-diffusion'});
% y = [size(keeped_C_vector(idx==2),2)/size(keeped_C_vector,2),size(keeped_C_vector(idx==3),2)/size(keeped_C_vector,2),size(keeped_C_vector(idx==1),2)/size(keeped_C_vector,2)];
b = bar(x,mean_proportions,'FaceColor','flat')
hold on
b.CData(3,:) = [0 0.4470 0.7410];
b.CData(2,:) = [0.9290 0.6940 0.1250];
b.CData(1,:) = [[0.8500 0.3250 0.0980]];
% ylim([0 1])
errorbar(mean_proportions, std_proportions, 'k', 'linestyle', 'none','linewidth',2); % Add error bars

ylabel('Frequency')
fig =gcf
ax =gca
ax.LineWidth = 2
set(gca,'FontSize',20,'FontName', 'Arial')
%% Histgram of one population
figure('Position', [400 200 1024 768]);
histogram(keeped_C_vector,edges)
edges = [0.0:0.01:0.5]
% histogram(Hall,edges, 'FaceColor','cyan')
xlabel('Hurst exponent');
ylabel('Counts');
title('NOGO-B,Diffusion coefficient');
% legend('Population 1', 'Population 2');
% legend('location','Northeast')
% legend box off
ax =gca
ax.LineWidth = 3
set(gca,'FontSize',28,'FontName', 'Arial','fontweight','bold')


%% diffusion coefficient based on three gaussian fitting
figure('Position', [400 200 1024 768]);
edges = [0.0:0.015:0.3]
population1_realunit = population1*0.58;
population2_realunit = population2*0.58;
population3_realunit = population3*0.58;
% histogram(Hall,edges, 'FaceColor','cyan')
histogram(population3_realunit,edges,'FaceColor', mixgaussiancmap(3,:), 'FaceAlpha', 1)
hold on
histogram(population1_realunit,edges,'FaceColor', mixgaussiancmap(1,:), 'FaceAlpha', 1)
hold on
histogram(population2_realunit,edges,'FaceColor', mixgaussiancmap(2,:), 'FaceAlpha', 1)

mean_d_all = [mean(population1),mean(population2),mean(population3)];
% histogram(keeped_sup_c,edges)
xlabel('Generalized diffusion coefficient');
ylabel('Counts');
title('NOGO-B,Hurst exponent');
% legend('Population 1', 'Population 2');
% legend('location','Northeast')
% legend box off
ax =gca
ax.LineWidth = 2
set(gca,'FontSize',28,'FontName', 'Arial','fontweight','bold')


% figure('Position', [400 200 1024 768]);
% plot(Hall,Call,'.','MarkerSize',15)
xlabel('D (\mum^{2}/s^{2H})');
ylabel('Counts');
% title('Generalized diffusion coefficient');
legend('Brownian motion','Super-diffusion', 'Sub-diffusion');
% legend('location','Northeast')
% legend box off
ax =gca
ax.LineWidth = 2
set(gca,'FontSize',28,'FontName', 'Arial','fontweight','bold')
legend box off
legend('location','northeast')

%% plot the 2D plot
% data = [keeped_C_vector;keeped_H_vector]';
% plot(H_all_f, C_all_f,'.','MarkerSize',20)
xlabel('Hurst exponent');
ylabel('Diffusion coefficient');
% title('SA-AF647 on SLB');
ax =gca
ax.LineWidth = 3
axis equal
% xlim([min(X(:)) max(X(:))]) % Make axes have the same scale
% ylim([min(Y(:)) max(Y(:))])
set(gca,'FontSize',28,'FontName', 'Arial','fontweight','bold')

% fit for gaussian mixture model
options = statset('Display','final');
GMModel = fitgmdist(data,2,'CovarianceType','diagonal','Options',options);

idx = cluster(GMModel, data);

% Scatter plot of the points with different colors for different clusters
figure;
hold on;
scatter( data(idx==1, 1),  data(idx==1, 2),'x', 'r','LineWidth',2);
scatter( data(idx==2, 1),  data(idx==2, 2),'x', 'b','LineWidth',2);
% scatter( data(idx==3, 1),  data(idx==3, 2),'x', 'g','LineWidth',2);
% scatter( data(idx==4, 1),  data(idx==4, 2),'x', 'c','LineWidth',2);
% scatter( data(idx==5, 1),  data(idx==5, 2),'x', 'y','LineWidth',2);
% Generate a grid of points to evaluate the PDF
x_range = linspace(min(data(:,1)), max(data(:,1)), 5);
y_range = linspace(min(data(:,2)), max(data(:,2)), 105);
[X, Y] = meshgrid(x_range, y_range);
grid_points = [X(:), Y(:)];

% Evaluate the PDF of the fitted GMM
pdf_values = pdf(GMModel, grid_points);

% Reshape the PDF values to match the grid
Z = reshape(pdf_values, length(y_range), length(x_range));

% Plot the Gaussian contours
contour(X, Y, Z, 'LineWidth', 2);

% Add labels and title
xlabel('Hurst exponent');
ylabel('Diffusion coefficient');
title('Gaussian Mixture Model for two populations');
legend('Population 1', 'Population 2');
hold off;

%%
figure('Position', [400 200 768 512]);
data = [keeped_H_vector;keeped_C_vector]';
x_range = linspace(min(data(:,1))-0.01, max(data(:,1))+0.01, 1000);
y_range = linspace(min(data(:,2))-0.01, max(data(:,2))+0.01, 1000);
[X, Y] = meshgrid(x_range, y_range);
grid_points = [X(:), Y(:)];
options = statset('Display','final');
GMModel = fitgmdist(data,2,'CovarianceType','diagonal','Options',options);
idx = cluster(GMModel, data);

% sigma4 = [squeeze(GMModel.Sigma(1,1,4)),0;0,squeeze(GMModel.Sigma(1,2,4))];
% sigma3 = [squeeze(GMModel.Sigma(1,1,3)),0;0,squeeze(GMModel.Sigma(1,2,3))];
sigma2 = [squeeze(GMModel.Sigma(1,1,2)),0;0,squeeze(GMModel.Sigma(1,2,2))];
sigma1 = [squeeze(GMModel.Sigma(1,1,1)),0;0,squeeze(GMModel.Sigma(1,2,1))];
gauss1 = mvnpdf(grid_points, GMModel.mu(1,:), sigma1);
gauss2 = mvnpdf(grid_points, GMModel.mu(2,:), sigma2);
% gauss3 = mvnpdf(grid_points, GMModel.mu(3,:), sigma3);
% gauss4 = mvnpdf(grid_points, GMModel.mu(4,:), sigma4);

pdf1 = reshape(gauss1, size(X));
pdf2 = reshape(gauss2, size(X));
% pdf3 = reshape(gauss3, size(X));
% pdf4 = reshape(gauss4, size(X));

combined_pdf = pdf1./max(max(pdf1)) + pdf2./max(max(pdf2))%+pdf3./max(max(pdf3))+pdf4./max(max(pdf4));

% Plot the combined PDF as a background
img=imagesc(x_range, y_range, combined_pdf);

set(gca,'YDir','normal') 
c = turbo(1200);
colormap(c(1:1000,:));
hold on
scatter( data(idx==1, 1),  data(idx==1, 2),[], 'r','x','LineWidth',2);
scatter( data(idx==2, 1),  data(idx==2, 2),[], 'g','x','LineWidth',2);
scatter( data(idx==3, 1),  data(idx==3, 2),[], 'b','x','LineWidth',2);
scatter( data(idx==4, 1),  data(idx==4, 2),[], 'c','x','LineWidth',2);
xlabel('Hurst exponent');
ylabel('Diffusion coefficient');
title('SA-AF647 diffuse on SLB');
legend('Population 1', 'Population 2');
legend('location','Northwest')
legend box off
hold off;

% imagesc(timelapsedata(:,:,data_num))
%% data format transfer for linux pc trained model
% load('K:\SPT_2023\Multi-objects tracking\Attention_SPTnet\model1_ourTIRF_linux4\experimentaldata_result_ER\FOV1_560__000_001.mat_cropped_all_subregion_2.mat')
load('K:\SPT_2023\Multi-objects tracking\Attention_SPTnet\model1_ourTIRF_linux4\experimentaldata_result_ER\FOV2_560__000_001.mat_cropped_all_subregion_3.mat')

estimation_xy_scale = (estimation_xy*32+32)*20;
estimation_C = estimation_C*0.5; 
% data format transfer for linux pc trained model
estimation_xy_perm = permute(estimation_xy_scale,[1,3,2,4]);
obj_estimation = permute(obj_estimation,[1,4,3,2]);

num_queries = 20;
colormap = jet(num_queries);
frmlist = 1:30;
datafilename = 3;
for data_num =1:1
    video_output = VideoWriter([folder,'/merge_SR_',num2str(datafilename),'.avi']); % Replace 'output_video.avi' with your desired output video file name
    video_output.FrameRate = 5; % Set the frame rate (frames per second)
    open(video_output);
    threshold = 0.9;
    frame_num = 30;
    cropsize =5;
    d = cropsize/2;
    predict = squeeze(obj_estimation(data_num,:,:)>threshold);
    C_all = [];
    H_all = [];
    for frame = 1:30
%         minpixel = min(SR_subregion(datafilename,:,:),[],'all');
%         maxpixel = max(SR_subregion(datafilename,:,:),[],'all');
%         frm=uint8((squeeze((SR_subregion(data_num,:,:))-minpixel)/(maxpixel-minpixel))*255);
%         rgb_img = gray2rgb(frm);
%         imshow(rgb_img,'InitialMagnification',100)

        dipshow(squeeze(SR_subregion(datafilename,:,:)))
        diptruesize(75)
        hold on
        count = 0;
        total_tracks = sum(sum(obj_estimation(1,:,:)>=threshold,2)>15);
        track_num = 0;
        for i = 1:num_queries
            cmap = parula(20);
            if obj_estimation(data_num,frame,i)>=threshold & sum(obj_estimation(data_num,:,i)>=threshold)>5
                track_num = track_num+1;
                plot(estimation_xy_perm(data_num,[frmlist(predict(:,i))],i,1)+1,estimation_xy_perm(data_num,[frmlist(predict(:,i))],i,2)+1,'-o','MarkerSize',1,'color',cmap(i,:),'LineWidth',1,'MarkerFaceColor',cmap(i,:))
                hold on
                for jj = 1:frame_num
                    cropbox(:,:,jj,i) = [estimation_xy_perm(data_num,jj,i,1)-d:estimation_xy_perm(data_num,jj,i,1)+d; estimation_xy_perm(data_num,jj,i,2)-d:estimation_xy_perm(data_num,jj,i,2)+d];
                %cropboxt(:,:,i) = [xt(i)-d:xt(i)+d; yt(i)-d:yt(i)+d];
                end
                scatter(estimation_xy_perm(data_num,frame,i,1)+1,estimation_xy_perm(data_num,frame,i,2)+1,100,cmap(i,:),'o','LineWidth',1)
%                 rectangle('position',[cropbox(1,1,frame,i)+1,cropbox(2,1,frame,i)+1,cropsize,cropsize],'EdgeColor',cmap(i,:),'LineWidth',2)
                text(estimation_xy_perm(data_num,frame,i,1)+30,estimation_xy_perm(data_num,frame,i,2)+20,['H:',num2str(round(estimation_H(data_num,i),2),'%.2f')],'Color',cmap(i,:),'fontsize',16,'FontWeight', 'bold')
                text(estimation_xy_perm(data_num,frame,i,1)+30,estimation_xy_perm(data_num,frame,i,2),['C:',num2str(round(estimation_C(data_num,i),2),'%.2f')],'Color',cmap(i,:),'fontsize',16,'FontWeight', 'bold') %cmap(i,:)
%               text(estimation_xy_perm(data_num,frame,i,1)+1-(cropsize-1)/2,estimation_xy_perm(data_num,frame,i,2)-(cropsize-1)/2,['H=',num2str(round(estimation_H(data_num,i),2)),'D=',num2str(round(estimation_C(data_num,i),2))],'Color','r','Clipping','on')
                hold on
                if estimation_H(data_num,i)~=0
                    count = count+1;
                    C_all(count) = round(estimation_C(data_num,i),2);
                    H_all(count) = round(estimation_H(data_num,i),2);
                end
            end
        end
        overlay_frame = getframe(gcf);
        writeVideo(video_output, overlay_frame);
    end
    close(video_output);    
end
%% Transfer gray scale image back to RGB format
function [Image]=gray2rgb(Image)
    %Gives a grayscale image an extra dimension
    %in order to use color within it
    [m n]=size(Image);
    rgb=zeros(m,n,3);
    rgb(:,:,1)=Image;
    rgb(:,:,2)=rgb(:,:,1);
    rgb(:,:,3)=rgb(:,:,1);
    Image=rgb/255;
end

%%
function [uniqueTracks,num_repeat,H_keeped_f,C_keeped_f] = removeRepeatedTracks(cellTracks, numOverlapSteps, distThreshold,H_keeped,C_keeped)
    % numOverlapSteps: Minimum number of overlapping steps to consider a track as repeated
    % distThreshold: Distance threshold to consider two points as overlapping
    uniqueTracks = cell(size(cellTracks));
    H_keeped_f = cell(size(H_keeped));
    C_keeped_f = cell(size(C_keeped));
    num_repeat = 0;
    for i = 1:size(cellTracks, 1)
        tracks = cellTracks(i, :);
        H_temp = H_keeped(i,:);
        C_temp = C_keeped(i,:);
        uniqueTracksRow = {};
        unique_H = {};
        unique_C = {};
        for j = 1:length(tracks)
            if isempty(tracks{j})
                continue;
            end
            isUnique = true;
            for k = 1:length(uniqueTracksRow)
                if isempty(uniqueTracksRow{k})
                    continue;
                end
                if isRepeatedTrack(tracks{j}, uniqueTracksRow{k}, numOverlapSteps, distThreshold)
                    isUnique = false;
                    num_repeat = num_repeat+1;
                    break;
                end
            end
            if isUnique
                uniqueTracksRow{end+1} = tracks{j}; 
                unique_H{end+1} = H_temp{j};
                unique_C{end+1} = C_temp{j};
            end
        end
        uniqueTracks(i, 1:length(uniqueTracksRow)) = uniqueTracksRow;
        H_keeped_f(i, 1:length(unique_H)) = unique_H;
        C_keeped_f(i, 1:length(unique_C)) = unique_C;
    end
end

function isRepeated = isRepeatedTrack(track1, track2, numOverlapSteps, distThreshold)
    % Calculate the number of overlapping steps and check the distance threshold
    steps1 = size(track1, 1);
    steps2 = size(track2, 1);

    overlapCount = 0;
    for i = 1:min(steps1, steps2)
        dist = sqrt((track1(i, 1) - track2(i, 1))^2 + (track1(i, 2) - track2(i, 2))^2);
        if dist <= distThreshold
            overlapCount = overlapCount + 1;
        end
        if overlapCount >= numOverlapSteps
            isRepeated = true;
            return;
        end
    end
    isRepeated = false;
end