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
%
% Updated March, 5th, 2026
% - Clip GT/inferenced trajectories within FOV (hide out of FOV ground truth)
% - Frame-by-frame trajectory display (only up to current frame)
% - Play/Pause autoplay + Step button (timer-based)
% - CSV output including Hurst exponent and diffusion coefficient and trajectories counting.
%%
SPTnetVisualizationGUI

function SPTnetVisualizationGUI
% SPTnetVisualizationGUI - GUI to visualize SPTnet inference or TIFF results
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

% Play / Step buttons
btnPlay = uicontrol('Parent',ctrlPanel,'Style','togglebutton','String','Play', ...
    'Position',[890,55,80,30],'Callback',@onPlayToggle);
uicontrol('Parent',ctrlPanel,'Style','pushbutton','String','Next Frame', ...
    'Position',[980,55,80,30],'Callback',@onStepOnce);

% Row 3: Frame slider
slider = uicontrol('Parent',ctrlPanel,'Style','slider','Min',1,'Max',30, ...
    'Value',1,'SliderStep',[1/29,1/29],'Position',[10,15,1080,20],        ...
    'Enable','off','Callback',@onSlide);
txtF = uicontrol('Parent',ctrlPanel,'Style','text','Position',[500,0,100,25], ...
    'String','Frame: 1','FontSize',12,'FontWeight','bold');

% Data and settings
data = struct();
setts.N = 20;
setts.T = 0.90;
setts.playFPS = 10;     % autoplay speed
data.current_idx = 1;
data.playTimer = [];

% Ensure timer cleanup on close
gfig.CloseRequestFcn = @onCloseGUI;

%% Callback implementations
function onLoadGT(~,~)
    [f,p] = uigetfile('*.mat','Select Ground Truth MAT'); if isequal(f,0), return; end
    gt = load(fullfile(p,f));

    % Detect image size from GT video
    [~, imgW] = size(gt.timelapsedata(:,:,1,1));
    scale = imgW / 2;

    % rescale GT coordinates if they are in normalized form
    if isfield(gt, 'traceposition')
        for i = 1:numel(gt.traceposition)
            tp = gt.traceposition{i};
            gt.traceposition{i} = tp + scale;
        end
    end

    data.GT = gt;

    % only enable/check it when traceposition exists
    if isfield(gt, 'traceposition')
        chkGT.Enable = 'on';
        chkGT.Value  = 1;
    else
        chkGT.Enable = 'off';
        chkGT.Value  = 0;
    end

    initSampleSelector(); enableSlider();
    slider.Value = 1; txtF.String = 'Frame: 1';
    updateDraw();
end

function onLoadINF(~,~)
    [f,p] = uigetfile('*.mat','Select Inference MAT'); if isequal(f,0), return; end
    S = load(fullfile(p,f));

    % Determine image size from TIFF or GT if available
    if isfield(data,'TIFF')
        [~, imgW] = size(data.TIFF(:,:,1));
    elseif isfield(data,'GT')
        [~, imgW] = size(data.GT.timelapsedata(:,:,1,1));
    else
        imgW = 64;  % fallback default
        warning('Image size not detected. Using default 64x64.');
    end

    % Assume square images, use width for scaling
    scale = imgW / 2;

    % Keep original permutation conventions
    data.INF.est_xy = permute(S.estimation_xy * scale + scale, [1,3,2,4]);
    data.INF.H      = S.estimation_H;
    data.INF.C      = S.estimation_C * 0.5;
    tmp_obj = permute(S.obj_estimation,[1,4,3,2]);
    data.INF.obj = tmp_obj(:,:,:,1);

    % Auto-detect Num Query from the second dimension of estimation_C
    if isfield(S, 'estimation_C')
        setts.N = size(S.estimation_C, 2);
        edtN.String = num2str(setts.N);
    end

    initSampleSelector(); enableSlider();
    slider.Value = 1; txtF.String = 'Frame: 1';
    updateDraw();
end

function onLoadTiff(~,~)
    [f,p] = uigetfile({'*.tif;*.tiff','TIFF Files'},'Select TIFF Video'); if isequal(f,0), return; end
    fname = fullfile(p,f); info = imfinfo(fname); nF = numel(info);
    firstFrame = imread(fname,1);
    vid = zeros(info(1).Height,info(1).Width,nF,class(firstFrame));
    for k = 1:nF
        vid(:,:,k) = imread(fname,k);
    end
    data.TIFF = vid;

    chkGT.Value = 0; chkGT.Enable = 'off';
    initSampleSelector(); enableSlider();
    slider.Value = 1; txtF.String = 'Frame: 1';
    updateDraw();
end

function onReset(~,~)
    stopPlayTimer();
    data = struct();
    setts.N = 20; setts.T = 0.90; setts.playFPS = 10;
    data.current_idx = 1; data.playTimer = [];

    chkGT.Value = 0; chkGT.Enable = 'off';
    popData.Enable = 'off'; popData.String = {'1'}; popData.Value = 1;
    edtN.String = num2str(setts.N);
    edtT.String = num2str(setts.T);

    slider.Enable = 'off'; slider.Value = 1;
    txtF.String = 'Frame: 1';

    if isvalid(btnPlay)
        btnPlay.Value = 0; btnPlay.String = 'Play';
    end

    cla(ax); axis(ax,'off');
end

function onSaveVideo(~,~)
    [fn,pn] = uiputfile('*.avi','Save Video As'); if isequal(fn,0), return; end
    out = fullfile(pn,fn);

    stopPlayTimer();
    if isvalid(btnPlay), btnPlay.Value = 0; btnPlay.String = 'Play'; end

    vw = VideoWriter(out,'Motion JPEG AVI');
    vw.FrameRate = 5;
    vw.Quality   = 100;
    open(vw);

    mf = round(slider.Max);
    origVal = slider.Value;

    for k = 1:mf
        slider.Value = k;
        txtF.String = ['Frame: ' num2str(k)];
        updateDraw();
        pos = get(ax,'Position');
        F = getframe(gfig,[pos(1),pos(2),pos(3),pos(4)]);
        writeVideo(vw,F);
    end

    slider.Value = origVal;
    txtF.String = ['Frame: ' num2str(round(origVal))];
    updateDraw();

    close(vw);
    msgbox('Video saved successfully','Success');
end

function onExportCSV(~,~)
    if ~isfield(data,'INF')
        errordlg('Load inference data first','Error');
        return;
    end

    idx = data.current_idx;
    nQ  = min(setts.N, size(data.INF.est_xy,3));
    nF  = round(slider.Max);

    % ---- Build "valid trajectory" list (track IDs) ----
    minLen = 5;  % minimum number of frames above threshold to count as a trajectory
    validQueries = false(1,nQ);
    for q = 1:nQ
        tr = (data.INF.obj(idx,1:nF,q) > setts.T);
        if sum(tr) >= minLen
            validQueries(q) = true;
        end
    end

    qList = find(validQueries);
    nTracks = numel(qList);

    if nTracks == 0
        warndlg(sprintf('No trajectories found (need >= %d frames above threshold). Try lowering threshold or minLen.', minLen), ...
            'Empty export');
        return;
    end

    % ---- Assemble rows ----
    % Columns: track_id, frame, x, y, Hurst, Diff.Coeffi, query_id
    rows = [];

    for t = 1:nTracks
        q = qList(t);  % original query index

        Hq = data.INF.H(1,q);
        Cq = data.INF.C(1,q);

        for f = 1:nF
            if data.INF.obj(idx,f,q) > setts.T
                xy = squeeze(data.INF.est_xy(idx,f,q,:));
                x = xy(1);
                y = xy(2);

                % track_id = t (1..nTracks), query_id = q 
                rows(end+1,:) = [t, f, x, y, Hq, Cq, q]; 
            end
        end
    end

    if isempty(rows)
        warndlg('No points above threshold (after filtering). Try lowering threshold.','Empty export');
        return;
    end

    % ---- Output table ----
    T = array2table(rows, 'VariableNames', ...
        {'track_ID','frame','x','y','Hurst exponent','Generalized diffusion coefficient','query_ID'});

    % by track then frame
    T = sortrows(T, {'track_ID','frame'});

    % ---- Save ----
    [fn,pn] = uiputfile('*.csv','Save CSV As');
    if isequal(fn,0), return; end
    writetable(T, fullfile(pn,fn));

    msgbox(sprintf('CSV exported.\nTotal trajectories: %d', nTracks), 'Success');
end

function onSelectData(src,~)
    data.current_idx = src.Value;
    enableSlider();
    slider.Value = 1;
    txtF.String = 'Frame: 1';
    updateDraw();
end

function onToggleGT(~,~)
    updateDraw();
end

function onChangeN(src,~)
    v = round(str2double(src.String));
    if isnan(v) || v < 1
        src.String = num2str(setts.N);
        return;
    end
    setts.N = v;
    updateDraw();
end

function onChangeT(src,~)
    v = str2double(src.String);
    if isnan(v) || v < 0 || v > 1
        src.String = num2str(setts.T);
        return;
    end
    setts.T = v;
    updateDraw();
end

function onSlide(src,~)
    fr = round(src.Value);
    txtF.String = ['Frame: ' num2str(fr)];
    updateDraw();
end

% NEW: Play controls
function onPlayToggle(src,~)
    if src.Value == 1
        src.String = 'Pause';
        startPlayTimer();
    else
        src.String = 'Play';
        stopPlayTimer();
    end
end

function onStepOnce(~,~)
    stopPlayTimer();
    if isvalid(btnPlay)
        btnPlay.Value = 0;
        btnPlay.String = 'Play';
    end
    advanceOneFrame();
end

function startPlayTimer()
    stopPlayTimer(); % ensure only one timer exists
    data.playTimer = timer( ...
        'ExecutionMode','fixedRate', ...
        'Period', max(0.01, 1/setts.playFPS), ...
        'TimerFcn', @(~,~) advanceOneFrame() ...
    );
    start(data.playTimer);
end

function stopPlayTimer()
    if isfield(data,'playTimer') && ~isempty(data.playTimer) && isvalid(data.playTimer)
        stop(data.playTimer);
        delete(data.playTimer);
    end
    data.playTimer = [];
end

function advanceOneFrame()
    if strcmp(slider.Enable,'off'), return; end
    mf = round(slider.Max);
    fr = round(slider.Value) + 1;
    if fr > mf, fr = 1; end
    slider.Value = fr;
    txtF.String = ['Frame: ' num2str(fr)];
    updateDraw();
    drawnow limitrate;
end

function onCloseGUI(~,~)
    stopPlayTimer();
    delete(gfig);
end

%% Setup and enabling
function initSampleSelector()
    if isfield(data,'GT')
        n = size(data.GT.timelapsedata,4);
    elseif isfield(data,'INF')
        n = size(data.INF.est_xy,1);
    elseif isfield(data,'TIFF')
        n = size(data.TIFF,3);
    else
        return;
    end

    popData.Enable = 'on';
    popData.String = arrayfun(@num2str,1:n,'UniformOutput',false);
    popData.Value  = min(data.current_idx,n);
end

function enableSlider()
    mf = 1;
    if isfield(data,'GT'),   mf = max(mf, size(data.GT.timelapsedata,3)); end
    if isfield(data,'INF'),  mf = max(mf, size(data.INF.est_xy,2));       end
    if isfield(data,'TIFF'), mf = max(mf, size(data.TIFF,3));             end

    slider.Enable = 'on';
    slider.Min = 1;
    slider.Max = mf;

    if mf <= 1
        slider.SliderStep = [1 1];
        slider.Value = 1;
    else
        slider.SliderStep = [1/(mf-1), 1/(mf-1)];
        slider.Value = min(max(round(slider.Value),1),mf);
    end
    txtF.String = ['Frame: ' num2str(round(slider.Value))];
end

%% Draw function
function updateDraw()
    cla(ax);
    hold(ax,'on');

    fr  = round(slider.Value);
    idx = data.current_idx;

    % choose background
    if isfield(data,'GT')
        Iraw = uint8(rescale(data.GT.timelapsedata(:,:,fr,idx))*255);
        Idisp = mat2gray(Iraw);
    elseif isfield(data,'TIFF')
        Iraw = data.TIFF(:,:,fr);
        % flip horizontally then rotate 90° 
        Iraw = fliplr(Iraw);
        Iraw = rot90(Iraw,1);
        Idisp = mat2gray(Iraw);
    else
        hold(ax,'off');
        return;
    end

    [imgH, imgW] = size(Idisp);

    imshow(repmat(Idisp,[1,1,3]),'Parent',ax);
    axis(ax,'image','off');

    % ---------------- GT overlay (frame-by-frame + clip to FOV) ----------------
    if chkGT.Value && isfield(data,'GT')
        [~,nC] = size(data.GT.traceposition);

        for ii = 1:nC
            tp = data.GT.traceposition{idx,ii};
            if isempty(tp) || size(tp,1) < fr || isnan(tp(fr,1))
                continue;
            end

            % only up to current frame
            tp_fr = tp(1:fr,:);
            tp_fr = clipToFOV(tp_fr, imgW, imgH);

            plot(ax, tp_fr(:,1)+1, tp_fr(:,2)+1, '-r', 'LineWidth', 1);

            % current point: show only if inside FOV
            x0 = tp(fr,1)+1;
            y0 = tp(fr,2)+1;
            if x0 >= 1 && x0 <= imgW && y0 >= 1 && y0 <= imgH
                scatter(ax, x0, y0, 40, 'r', 'x', 'LineWidth', 1);
                text(ax, x0, y0-4, sprintf('H=%.2f,D=%.2f', ...
                    data.GT.Hlabel{idx,ii}, data.GT.Clabel{idx,ii}), ...
                    'Color','r','FontSize',10);
            end
        end
    end

    % ---------------- INF overlay (frame-by-frame accumulation + clip) ----------------
    if isfield(data,'INF')
        nQ   = min(setts.N, size(data.INF.est_xy,3));
        cmap = lines(nQ);

        for q = 1:nQ
            % active frames up to current frame
            activeUpToFr = data.INF.obj(idx,1:fr,q) > setts.T;

            % draw trajectory only up to current frame
            if sum(activeUpToFr) >= 2
                pts = squeeze(data.INF.est_xy(idx,1:fr,q,:)); % (fr x 2)
                pts(~activeUpToFr,:) = NaN;                  % break line where inactive
                pts = clipToFOV(pts, imgW, imgH);            % hide outside FOV
                plot(ax, pts(:,1)+1, pts(:,2)+1, '-', 'Color', cmap(q,:), 'LineWidth', 1);
            end

            % current-frame marker if active AND inside FOV
            if data.INF.obj(idx,fr,q) > setts.T
                pt = squeeze(data.INF.est_xy(idx,fr,q,:));
                x  = double(pt(1)) + 1;
                y  = double(pt(2)) + 1;

                if x < 1 || x > imgW || y < 1 || y > imgH
                    continue;
                end

                boxSize = 3;
                halfBox = boxSize/2;
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

%% Helper: clip/hide points outside FOV by setting them to NaN
function pts = clipToFOV(pts, imgW, imgH)
    if isempty(pts), return; end
    x = pts(:,1) + 1;  % display coordinates
    y = pts(:,2) + 1;
    out = (x < 1) | (x > imgW) | (y < 1) | (y > imgH);
    pts(out,:) = NaN;
end

end