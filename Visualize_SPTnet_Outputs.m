%%
% Main script for generating simulation videos for SPTnet training
%
% (C) Copyright 2024                The Huang Lab
%
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
%
%     Author: Cheng Bi, October 2024
%%
% Load test videos file
[file, path] = uigetfile('*.mat', 'Select the test data file');
fullFilePath = fullfile(path, file)
load(fullFilePath)
% Load SPTnet output files
[file, path] = uigetfile('*.mat', 'Select the SPTnet inference output file');
fullFilePath = fullfile(path, file)
load(fullFilePath)


% data format transfer 
savepath_visulization_result = pwd;
estimation_xy_scale = estimation_xy*32+32;
estimation_C = estimation_C*0.5; 
estimation_xy_perm = permute(estimation_xy_scale,[1,3,2,4]);
obj_estimation = squeeze(permute(obj_estimation,[1,4,3,2]));
%
num_queries = 20;
colormap = jet(num_queries);
frmlist = 1:30;

for data_num =1:5
    video_output = VideoWriter([savepath_visulization_result,'exampleoutput_',num2str(data_num),'.avi']); % Replace 'output_video.avi' with your desired output video file name
    video_output.FrameRate = 5; % Set the frame rate (frames per second)
    open(video_output);
    threshold = 0.90; % detection threshold
    frame_num = 30;
    cropsize =5;
    d = cropsize/2;
    predict = squeeze(obj_estimation(data_num,:,:)>threshold);
    for frame = 1:frame_num
        minpixel = min(timelapsedata(:,:,:,data_num),[],'all');
        maxpixel = max(timelapsedata(:,:,:,data_num),[],'all');
        frm=uint8(((timelapsedata(:,:,frame,data_num)-minpixel)/(maxpixel-minpixel))*255);
        rgb_img = gray2rgb(frm);
        imshow(rgb_img,'InitialMagnification',2000)
        hold on
        for ii = 1:sum(~cellfun('isempty', traceposition(data_num,:)))
            if  ~isnan(traceposition{data_num,ii}(frame,1))== 1
                gtlist = ~isnan(traceposition{data_num,ii}(:,1));
                scatter(traceposition{data_num,ii}(frame,1)+1,traceposition{data_num,ii}(frame,2)+1,200,'r','x','LineWidth',2)
                set(gca,'FontSize',14,'FontName', 'Arial','fontweight','bold')
    %             set(gcf,'Position',[700 50 1024 1024])
                plot(traceposition{data_num,ii}(frmlist(gtlist),1)+1,traceposition{data_num,ii}(frmlist(gtlist),2)+1,'-o','MarkerSize',2,'color','r','LineWidth',2,'MarkerFaceColor','r')                
                cropboxt = [traceposition{data_num,ii}(frame,1)-d+1:traceposition{data_num,ii}(frame,1)+d+1; traceposition{data_num,ii}(frame,2)-d+1:traceposition{data_num,ii}(frame,2)+d+1];
%                 rectangle('position',[cropboxt(1,1)+1,cropboxt(2,1)+1,cropsize,cropsize],'EdgeColor','r','LineWidth',2)
                text(cropboxt(1,1)-0.5*d,cropboxt(2,1)+2.5*d,['H=',num2str(round(Hlabel{data_num,ii},2),'%.2f'),','],'Color','r','fontsize',16)
                text(cropboxt(1,1)+1.5*d,cropboxt(2,1)+2.5*d,['D=',num2str(round(Clabel{data_num,ii},2),'%.2f')],'Color','r','fontsize',16)
            end
        end
        for i = 1:num_queries
            cmap = parula(num_queries);
            if obj_estimation(data_num,frame,i)>=threshold & sum(obj_estimation(data_num,:,i)>=threshold)>=5 % output tracks lasting more than 5 frames
                plot(estimation_xy_perm(data_num,[frmlist(predict(:,i))],i,1)+1,estimation_xy_perm(data_num,[frmlist(predict(:,i))],i,2)+1,'-o','MarkerSize',2,'color',cmap(i,:),'LineWidth',2,'MarkerFaceColor',cmap(i,:))
                hold on
                for jj = 1:frame_num
                    cropbox(:,:,jj,i) = [estimation_xy_perm(data_num,jj,i,1)-d:estimation_xy_perm(data_num,jj,i,1)+d; estimation_xy_perm(data_num,jj,i,2)-d:estimation_xy_perm(data_num,jj,i,2)+d];
                %cropboxt(:,:,i) = [xt(i)-d:xt(i)+d; yt(i)-d:yt(i)+d];
                end
                scatter(estimation_xy_perm(data_num,frame,i,1)+1,estimation_xy_perm(data_num,frame,i,2)+1,200,cmap(i,:),'o','LineWidth',2)
                rectangle('position',[cropbox(1,1,frame,i)+1,cropbox(2,1,frame,i)+1,cropsize,cropsize],'EdgeColor',cmap(i,:),'LineWidth',2)
                text(cropbox(1,1,frame,i)-0.5*d,cropbox(2,1,frame,i)-0.2*d,['H=',num2str(round(estimation_H(data_num,i),2),'%.2f'),','],'Color',cmap(i,:),'fontsize',16)
                text(cropbox(1,1,frame,i)+1.5*d,cropbox(2,1,frame,i)-0.2*d,['D=',num2str(round(estimation_C(data_num,i),2),'%.2f')],'Color',cmap(i,:),'fontsize',16)
                hold on
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