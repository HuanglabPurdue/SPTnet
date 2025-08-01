% Main script for visualizing SPTnet outputs
%
% (C) Copyright 2025                The Huang Lab
%
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
%
%     Author: Cheng Bi, July 2025
%%
SPTnetVisualizationGUI

function SPTnetVisualizationGUI
% SPTnetVisualizationGUI - GUI to visualize SPTnet inference or TIFF results
% Supports GT, INF, and TIFF-only videos; selectable samples; adjustable queries and threshold;
% video export (axes only); export CSV of inference tracks; reset functionality;

% Create main GUI figure
gfig = figure('Name','SPTnet Visualization','NumberTitle','off', ...
    'MenuBar','none','ToolBar','none','Position',[100,100,1200,900]);

% Axes for display
ax = axes('Parent',gfig,'Units','pixels','Position',[50,200,1100,600]); axis(ax,'off');
% Highlight video display region
pos_ax = get(ax,'Position');
pos_fig = get(gfig,'Position');
norm_pos = [pos_ax(1)/pos_fig(3), pos_ax(2)/pos_fig(4), pos_ax(3)/pos_fig(3), pos_ax(4)/pos_fig(4)];
annotation(gfig,'rectangle',norm_pos,'Color','black','LineWidth',2);

% Control panel at bottom
ctrlPanel = uipanel('Parent',gfig,'Units','pixels','Position',[50,50,1100,130], ...
    'BorderType','none');

% Row 1: Load and reset buttons
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Load ground truth',   ...
    'Position',[10,85,100,30],'Callback',@onLoadGT);
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Load SPTnet output',  ...
    'Position',[120,85,100,30],'Callback',@onLoadINF);
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Load Tiff', ...
    'Position',[230,85,100,30],'Callback',@onLoadTiff);
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Reset',     ...
    'Position',[340,85,100,30],'Callback',@onReset);

% Row 2: Display options and parameters
chkGT = uicontrol('Parent',ctrlPanel,'Style','checkbox','String','Show GT', ...
    'Value',1,'Enable','off','Position',[10,55,100,30],'Callback',@onToggleGT);
uicontrol('Parent',ctrlPanel,'Style','text','String','Sample:',        ...
    'Position',[120,60,60,20]);
popData = uicontrol('Parent',ctrlPanel,'Style','popupmenu','String',{'1'}, ...
    'Enable','off','Position',[190,55,80,25],'Callback',@onSelectData);

uicontrol('Parent',ctrlPanel,'Style','text','String','Num Query:',        ...
    'Position',[290,60,60,20]);
edtN = uicontrol('Parent',ctrlPanel,'Style','edit','String','20',     ...
    'Position',[360,55,80,25],'Callback',@onChangeN);
uicontrol('Parent',ctrlPanel,'Style','text','String','Threshold:',    ...
    'Position',[450,60,80,20]);
edtT = uicontrol('Parent',ctrlPanel,'Style','edit','String','0.90',   ...
    'Position',[540,55,80,25],'Callback',@onChangeT);

uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Save Video', ...
    'Position',[630,55,120,30],'Callback',@onSaveVideo);
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Export CSV',  ...
    'Position',[760,55,120,30],'Callback',@onExportCSV);

% Row 3: Frame slider
slider = uicontrol('Parent',ctrlPanel,'Style','slider','Min',1,'Max',30, ...
    'Value',1,'SliderStep',[1/29,1/29],'Position',[10,15,1080,20],        ...
    'Enable','off','Callback',@onSlide);
txtF = uicontrol('Parent',ctrlPanel,'Style','text','Position',[500,0,100,25], ...
    'String','Frame: 1','FontSize',12,'FontWeight','bold');

% Data and settings
data = struct(); setts.N = 20; setts.T = 0.90; data.current_idx = 1;

%% Callback implementations
function onLoadGT(~,~)
    [f,p] = uigetfile('*.mat','Select Ground Truth MAT'); if isequal(f,0), return; end
    gt = load(fullfile(p,f));

    % Detect image size from GT video
    [imgH, imgW] = size(gt.timelapsedata(:,:,1,1));
    scale = imgW / 2;

    % Optional: rescale GT coordinates if they are in normalized form
    % Skip if already in pixel units
    if isfield(gt, 'traceposition')
        for i = 1:numel(gt.traceposition)
            tp = gt.traceposition{i};
            gt.traceposition{i} = tp + scale;
        end
    end

    data.GT = gt;
    chkGT.Enable = 'on'; chkGT.Value = 1;
    initSampleSelector(); enableSlider(); updateDraw();
end

function onLoadINF(~,~)
    [f,p] = uigetfile('*.mat','Select Inference MAT'); if isequal(f,0), return; end
    S = load(fullfile(p,f));

    % Determine image size from TIFF or GT if available
    if isfield(data,'TIFF')
        [imgH, imgW] = size(data.TIFF(:,:,1));
    elseif isfield(data,'GT')
        [imgH, imgW] = size(data.GT.timelapsedata(:,:,1,1));
    else
        imgH = 64; imgW = 64;  % fallback default
        warning('Image size not detected. Using default 64x64.');
    end

    % Assume square images, use width for scaling
    scale = imgW / 2;
    data.INF.est_xy = permute(S.estimation_xy * scale + scale, [1,3,2,4]);
    data.INF.H      = S.estimation_H;
    data.INF.C      = S.estimation_C * 0.5;
    tmp_obj = permute(S.obj_estimation,[1,4,3,2]);
    data.INF.obj = tmp_obj(:,:,:,1);

    initSampleSelector(); enableSlider(); updateDraw();
end

function onLoadTiff(~,~)
    [f,p] = uigetfile({'*.tif;*.tiff','TIFF Files'},'Select TIFF Video'); if isequal(f,0), return; end
    fname = fullfile(p,f); info = imfinfo(fname); nF = numel(info);
    firstFrame = imread(fname,1);
    vid = zeros(info(1).Height,info(1).Width,nF,class(firstFrame));
    for k = 1:nF, vid(:,:,k) = imread(fname,k); end
    data.TIFF = vid;
    chkGT.Value = 0; chkGT.Enable = 'off'; initSampleSelector(); enableSlider(); updateDraw();
end

function onReset(~,~)
    data = struct(); setts.N = 20; setts.T = 0.90; data.current_idx = 1;
    chkGT.Value = 0; chkGT.Enable = 'off'; popData.Enable = 'off'; popData.String = {'1'}; popData.Value = 1;
    edtN.String = num2str(setts.N); edtT.String = num2str(setts.T);
    slider.Enable = 'off'; slider.Value = 1; txtF.String = 'Frame: 1'; cla(ax); axis(ax,'off');
end

function onSaveVideo(~,~)
    [fn,pn] = uiputfile('*.avi','Save Video As'); if isequal(fn,0), return; end
    out = fullfile(pn,fn); vw = VideoWriter(out,'Motion JPEG AVI'); vw.FrameRate=5; vw.Quality=100; open(vw);
    mf = round(slider.Max);
    for k = 1:mf
        slider.Value = k; txtF.String = ['Frame: ' num2str(k)]; updateDraw();
        pos = get(ax,'Position'); F = getframe(gfig,[pos(1),pos(2),pos(3),pos(4)]);
        writeVideo(vw,F);
    end
    close(vw); msgbox('Video saved successfully','Success');
end

function onExportCSV(~,~)
    if ~isfield(data,'INF'), errordlg('Load inference data first','Error'); return; end
    idx = data.current_idx; nQ = min(setts.N,size(data.INF.est_xy,3)); nF = round(slider.Max);
    rows = [];
    for q = 1:nQ
        for f = 1:nF
            if data.INF.obj(idx,f,q) > setts.T
                xy = data.INF.est_xy(idx,f,q,:); x = xy(1); y = xy(2);
                rows(end+1,:) = [q,f,x,y];
            end
        end
    end
    T = array2table(rows,'VariableNames',{'particle_ID','frame','x','y'});
    [fn,pn] = uiputfile('*.csv','Save CSV As'); if isequal(fn,0), return; end
    writetable(T,fullfile(pn,fn)); msgbox('CSV exported','Success');
end

function onSelectData(src,~), data.current_idx = src.Value; enableSlider(); updateDraw(); end
function onToggleGT(~,~), updateDraw(); end

function onChangeN(src,~)
    v = round(str2double(src.String)); if isnan(v)||v<1, src.String = num2str(setts.N); return; end
    setts.N = v; updateDraw();
end

function onChangeT(src,~)
    v = str2double(src.String); if isnan(v)||v<0||v>1, src.String = num2str(setts.T); return; end
    setts.T = v; updateDraw();
end

function onSlide(src,~)
    fr = round(src.Value); txtF.String = ['Frame: ' num2str(fr)]; updateDraw();
end

%% Setup and enabling
function initSampleSelector()
    if isfield(data,'GT'),       n = size(data.GT.timelapsedata,4);
    elseif isfield(data,'INF'),   n = size(data.INF.est_xy,1);
    elseif isfield(data,'TIFF'),  n = size(data.TIFF,3);
    else return; end
    popData.Enable = 'on'; popData.String = arrayfun(@num2str,1:n,'UniformOutput',false);
    popData.Value = min(data.current_idx,n);
end

function enableSlider()
    mf = 1;
    if isfield(data,'GT'),       mf = max(mf,size(data.GT.timelapsedata,3)); end
    if isfield(data,'INF'),      mf = max(mf,size(data.INF.est_xy,2));       end
    if isfield(data,'TIFF'),     mf = max(mf,size(data.TIFF,3));             end
    slider.Enable = 'on'; slider.Max = mf;
    slider.SliderStep = [1/(mf-1),1/(mf-1)]; txtF.String = ['Frame: ' num2str(round(slider.Value))];
end

%% Draw function
function updateDraw()
    cla(ax); hold(ax,'on'); fr = round(slider.Value); idx = data.current_idx;
    % choose background
    if isfield(data,'GT')
        Iraw = uint8(rescale(data.GT.timelapsedata(:,:,fr,idx))*255);
    elseif isfield(data,'TIFF')
        Iraw = data.TIFF(:,:,fr);
        % flip horizontally then rotate 90° CCW
        Iraw = fliplr(Iraw);
        Iraw = rot90(Iraw,1);
        Iraw = mat2gray(Iraw);
    else return; end
    imshow(repmat(Iraw,[1,1,3]),'Parent',ax); axis(ax,'image','off');

    % GT overlay
    if chkGT.Value && isfield(data,'GT')
        [~,nC] = size(data.GT.traceposition);
        for ii = 1:nC
            tp = data.GT.traceposition{idx,ii}; if isempty(tp)||size(tp,1)<fr||isnan(tp(fr,1)), continue; end
            valid = ~isnan(tp(:,1));
            plot(ax,tp(valid,1)+1,tp(valid,2)+1,'-r','LineWidth',1);
            scatter(ax,tp(fr,1)+1,tp(fr,2)+1,40,'r','x','LineWidth',1);
            text(ax,tp(fr,1)+1,tp(fr,2)-4,sprintf('H=%.2f,D=%.2f',data.GT.Hlabel{idx,ii},data.GT.Clabel{idx,ii}),'Color','r','FontSize',10);
        end
    end

    % inference overlay with colored square
    if isfield(data,'INF')
        nQ   = min(setts.N, size(data.INF.est_xy,3));
        cmap = lines(nQ);
        for q = 1:nQ
            tr = data.INF.obj(idx,:,q) > setts.T; if sum(tr) < 5, continue; end
            pts = squeeze(data.INF.est_xy(idx,tr,q,:));
            plot(ax, pts(:,1)+1, pts(:,2)+1, '-', 'Color', cmap(q,:), 'LineWidth',1);
            if data.INF.obj(idx,fr,q) > setts.T
                pt = squeeze(data.INF.est_xy(idx,fr,q,:));
                x  = double(pt(1)) + 1;
                y  = double(pt(2)) + 1;
                boxSize = 3;           % side length of square (3x3 pixels)
                halfBox = boxSize/2;           % half-length
                rectangle('Position',[x-halfBox, y-halfBox, boxSize, boxSize], ...
                          'EdgeColor', cmap(q,:), 'LineWidth', 1, 'Parent', ax);
                scatter(ax, x, y, 20, cmap(q,:), 'filled');
                text(ax, x, y-2, sprintf('H=%.2f,D=%.2f', data.INF.H(1,q), data.INF.C(1,q)), ...
                     'Color', cmap(q,:), 'FontSize',10);
            end
        end
    end
    hold(ax,'off');
end
end