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
% Updated October 12, 2026: add fBm parameter-trajectory compatibility and correct sample indexing.
%%
SPTnetVisualizationGUI

function SPTnetVisualizationGUI
% SPTnetVisualizationGUI - GUI to visualize SPTnet inference or TIFF results
% Keep the sibling add-in available after run(full_path) restores its caller folder.
addpath(fileparts(mfilename('fullpath')));
% Create main GUI figure
gfig = figure('Name','SPTnet Visualization','NumberTitle','off', ...
    'MenuBar','none','ToolBar','none','Position',[100,100,1200,900], ...
    'Tag','SPTnetVisualizationGUI');

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

% Check every query with positions above the GUI threshold.
btnConsistency = uicontrol('Parent',ctrlPanel,'Style','togglebutton', ...
    'String','Compatibility p-value: Off','Value',0,'Enable','off','Position',[450,85,220,30], ...
    'TooltipString','Monte Carlo compatibility of the reported H and D with the trajectory under an ideal fBm model.', ...
    'Tag','GenerativeConsistencyButton','Callback',@onGenerativeConsistency);
uicontrol('Parent',ctrlPanel,'Style','text','String','All tracks above threshold', ...
    'HorizontalAlignment','left','Position',[685,85,250,25],'Visible','off');
txtConsistencyStatus = uicontrol('Parent',gfig,'Style','text', ...
    'String','Load SPTnet output, set the threshold, then compute compatibility p-values.', ...
    'HorizontalAlignment','left','Position',[50,820,1100,35], ...
    'Tag','GenerativeConsistencyStatus','Visible','off');

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
setts.consistency = struct('Seed',20261012);
setts.showConsistency = false;

% Ensure timer cleanup on close
gfig.CloseRequestFcn = @onCloseGUI;

% Small public interface for reproducible integration tests and programmatic use.
addin = struct('loadInference',@loadInferenceFile, ...
    'loadGroundTruth',@(path)onLoadGT([],[],path),'loadTiff',@(path)onLoadTiff([],[],path), ...
    'selectSample',@selectConsistencySample,'getQueries',@getValidQueries, ...
    'getInput',@getConsistencyInput,'getResults',@getConsistencyResults, ...
    'run',@runGenerativeConsistency,'configure',@configureConsistency);
addin.handles = struct('button',btnConsistency,'axes',ax, ...
    'sample',popData,'threshold',edtT,'numQueries',edtN,'slider',slider, ...
    'status',txtConsistencyStatus);
setappdata(gfig,'SPTnetGenerativeAddin',addin);
setappdata(gfig,'SPTnetTrajectoryAddin',addin);

%% Callback implementations
function onLoadGT(~,~,filePath)
    if nargin<3
        [f,p] = uigetfile('*.mat','Select Ground Truth MAT'); if isequal(f,0), return; end
        filePath = fullfile(p,f);
    end
    gt = load(filePath);

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
    invalidateConsistency();

    % only enable/check it when traceposition exists
    if isfield(gt, 'traceposition')
        chkGT.Enable = 'on';
        chkGT.Value  = 1;
    else
        chkGT.Enable = 'off';
        chkGT.Value  = 0;
    end

    initSampleSelector(); enableSlider();
    initConsistencyControls();
    slider.Value = 1; txtF.String = 'Frame: 1';
    updateDraw();
end

function onLoadINF(~,~)
    [f,p] = uigetfile('*.mat','Select Inference MAT'); if isequal(f,0), return; end
    loadInferenceFile(fullfile(p,f));
end

function loadInferenceFile(inferencePath)
    S = load(inferencePath);

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
    data.INF.imageWidth = imgW;
    data.INF.sourcePath = inferencePath;
    invalidateConsistency();

    % Auto-detect Num Query from the second dimension of estimation_C
    if isfield(S, 'estimation_C')
        setts.N = size(S.estimation_C, 2);
        edtN.String = num2str(setts.N);
    end

    initSampleSelector(); enableSlider();
    initConsistencyControls();
    slider.Value = 1; txtF.String = 'Frame: 1';
    updateDraw();
end

function onLoadTiff(~,~,filePath)
    if nargin<3
        [f,p] = uigetfile({'*.tif;*.tiff','TIFF Files'},'Select TIFF Video'); if isequal(f,0), return; end
        filePath = fullfile(p,f);
    end
    fname = filePath; info = imfinfo(fname); nF = numel(info);
    firstFrame = imread(fname,1);
    vid = zeros(info(1).Height,info(1).Width,nF,class(firstFrame));
    for k = 1:nF
        vid(:,:,k) = imread(fname,k);
    end
    data.TIFF = vid;
    invalidateConsistency();

    chkGT.Value = 0; chkGT.Enable = 'off';
    initSampleSelector(); enableSlider();
    initConsistencyControls();
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
    btnConsistency.Enable = 'off';
    invalidateConsistency();
    txtConsistencyStatus.String = 'Load SPTnet output, set the threshold, then compute compatibility p-values.';
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

function onExportCSV(~,~,filePath)
    if ~isfield(data,'INF')
        errordlg('Load inference data first','Error');
        return;
    end

    idx = data.current_idx;
    nF  = size(data.INF.est_xy,2);

    % ---- Build "valid trajectory" list (track IDs) ----
    minLen = 5;  % minimum number of frames above threshold to count as a trajectory
    qList = getValidQueries(minLen);
    nTracks = numel(qList);

    if nTracks == 0
        warndlg(sprintf('No trajectories found (need >= %d frames above threshold). Try lowering threshold or minLen.', minLen), ...
            'Empty export');
        return;
    end

    % ---- Assemble rows ----
    % Columns: track_id, frame, x, y, Hurst, Diff.Coeffi, query_id, compatibility_p_value
    rows = [];
    batch = getConsistencyResults();

    for t = 1:nTracks
        q = qList(t);  % original query index

        Hq = data.INF.H(idx,q);
        Cq = data.INF.C(idx,q);
        % Export the cached trajectory p-value even when its display is Off.
        pValue = NaN; % not yet calculated or unsupported
        if ~isempty(batch) && q<=numel(batch.p_values) && batch.computed(q)
            pValue = batch.p_values(q);
        end

        for f = 1:nF
            if data.INF.obj(idx,f,q) > setts.T
                xy = squeeze(data.INF.est_xy(idx,f,q,:));
                x = xy(1);
                y = xy(2);

                % track_id = t (1..nTracks), query_id = q 
                rows(end+1,:) = [t, f, double(x), double(y), double(Hq), double(Cq), q, pValue];
            end
        end
    end

    if isempty(rows)
        warndlg('No points above threshold (after filtering). Try lowering threshold.','Empty export');
        return;
    end

    % ---- Output table ----
    T = array2table(rows, 'VariableNames', ...
        {'track_ID','frame','x','y','Hurst exponent','Generalized diffusion coefficient','query_ID', ...
         'compatibility_p_value'});

    % by track then frame
    T = sortrows(T, {'track_ID','frame'});

    % ---- Save ----
    if nargin<3
        [fn,pn] = uiputfile('*.csv','Save CSV As');
        if isequal(fn,0), return; end
        filePath = fullfile(pn,fn);
    end
    writetable(T, filePath);

    msgbox(sprintf('CSV exported.\nTotal trajectories: %d', nTracks), 'Success');
end

function onSelectData(src,~)
    data.current_idx = src.Value;
    setts.showConsistency = false;
    initConsistencyControls();
    enableSlider();
    slider.Value = 1;
    txtF.String = 'Frame: 1';
    updateDraw();
end

function onToggleGT(~,~)
    updateDraw();
end

%% Trajectory-only fBm parameter-trajectory compatibility
function initConsistencyControls()
    btnConsistency.Enable = 'off';
    btnConsistency.Value = double(setts.showConsistency);
    if setts.showConsistency
        btnConsistency.String = 'Compatibility p-value: On';
    else
        btnConsistency.String = 'Compatibility p-value: Off';
    end
    if ~isfield(data,'INF') || data.current_idx > size(data.INF.est_xy,1)
        return;
    end
    if exist('SPTnet_trajectory_compatibility','file') ~= 2
        txtConsistencyStatus.String = 'SPTnet_trajectory_compatibility.m is missing.';
        return;
    end
    btnConsistency.Enable = 'on';
    batch = getConsistencyResults();
    if isempty(batch)
        txtConsistencyStatus.String = sprintf('%d tracks above threshold; turn compatibility p-values on to compute.',numel(getValidQueries()));
        clearConsistencyAppdata();
    else
        setappdata(gfig,'SPTnetGenerativeLastResult',batch);
        setappdata(gfig,'SPTnetTrajectoryLastResult',batch);
        showConsistencySummary(batch);
    end
end

function qList = getValidQueries(minLength)
    % One accepted position includes a query; short tracks are reported as N/A.
    % CSV retains its original five-frame rule by passing minLength = 5.
    if nargin<1, minLength = 1; end
    qList = [];
    if ~isfield(data,'INF') || data.current_idx > size(data.INF.est_xy,1), return; end
    nQ = min(setts.N,size(data.INF.est_xy,3));
    nF = size(data.INF.est_xy,2);
    active = reshape(data.INF.obj(data.current_idx,1:nF,1:nQ),nF,nQ) > setts.T;
    qList = find(sum(active,1)>=minLength);
end

function configureConsistency(options)
    names = fieldnames(options);
    proposed = setts.consistency;
    for ii = 1:numel(names)
        if ~isfield(proposed,names{ii})
            error('SPTnet:ConsistencySetup','Unknown check setting: %s',names{ii});
        end
        proposed.(names{ii}) = options.(names{ii});
    end
    validateattributes(proposed.Seed,{'numeric'},{'scalar','integer','>=',0,'<=',4294967295});
    setts.consistency = proposed;
    onConsistencySettingsChanged([],[]);
end

function onConsistencySettingsChanged(~,~)
    invalidateConsistency();
    initConsistencyControls();
    updateDraw();
end

function invalidateConsistency()
    data.consistency = {};
    setts.showConsistency = false;
    btnConsistency.Value = 0;
    btnConsistency.String = 'Compatibility p-value: Off';
    clearConsistencyAppdata();
end

function clearConsistencyAppdata()
    names = {'SPTnetGenerativeLastResult','SPTnetTrajectoryLastResult'};
    for ii = 1:numel(names)
        if isappdata(gfig,names{ii}), rmappdata(gfig,names{ii}); end
    end
end

function count = consistencyBootstrapCount()
    count = 199;
end

function batch = getConsistencyResults()
    batch = [];
    idx = data.current_idx;
    if ~isfield(data,'INF') || ~isfield(data,'consistency') || ...
            idx>numel(data.consistency) || isempty(data.consistency{idx})
        return;
    end
    saved = data.consistency{idx};
    if saved.threshold~=setts.T || saved.numQueries~=min(setts.N,size(data.INF.est_xy,3)) || ...
            saved.bootstrapCount~=consistencyBootstrapCount() || saved.seed~=setts.consistency.Seed || ...
            saved.imageWidth~=data.INF.imageWidth || ~strcmp(saved.sourcePath,data.INF.sourcePath)
        return;
    end
    batch = saved;
end

function selectConsistencySample(idx)
    validateattributes(idx,{'numeric'},{'scalar','integer','positive'});
    if ~isfield(data,'INF') || idx > size(data.INF.est_xy,1)
        error('SPTnet:ConsistencySample','Sample is outside the loaded inference data.');
    end
    popData.Value = idx;
    onSelectData(popData,[]);
end

function input = getConsistencyInput(q)
    if ~isfield(data,'INF')
        error('SPTnet:ConsistencyInput','Load SPTnet inference data first.');
    end
    validateattributes(q,{'numeric'},{'scalar','integer','positive'});
    idx = data.current_idx;
    if q>min(setts.N,size(data.INF.est_xy,3)) || idx>size(data.INF.est_xy,1)
        error('SPTnet:ConsistencyQuery','Query is outside the current sample or Num Query setting.');
    end
    nFrames = size(data.INF.est_xy,2);
    if isfield(data,'TIFF')
        currentImageWidth = size(data.TIFF,2);
    elseif isfield(data,'GT')
        currentImageWidth = size(data.GT.timelapsedata,2);
    else
        currentImageWidth = data.INF.imageWidth;
    end
    if currentImageWidth ~= data.INF.imageWidth
        error('SPTnet:GenerativeImageScaling', ...
            'Image size changed after inference was loaded. Reload SPTnet output to recalculate its coordinate scale.');
    end
    % Keep acquisition times when the threshold removes intermediate positions.
    active = reshape(data.INF.obj(idx,:,q),nFrames,1) > setts.T;
    xy = reshape(data.INF.est_xy(idx,:,q,:),nFrames,2);
    input = struct('xy',xy(active,:),'H',data.INF.H(idx,q),'D',data.INF.C(idx,q), ...
        'frames',find(active),'sample',idx,'query',q,'threshold',setts.T, ...
        'imageWidth',data.INF.imageWidth,'sourcePath',data.INF.sourcePath);
end

function onGenerativeConsistency(~,~)
    try
        runGenerativeConsistency();
    catch err
        if isgraphics(gfig), errordlg(err.message,'fBm parameter-trajectory compatibility'); end
    end
end

function batch = runGenerativeConsistency()
    if ~isfield(data,'INF')
        error('SPTnet:ConsistencyInput','Load SPTnet inference data first.');
    end
    batch = getConsistencyResults();
    if ~isempty(batch) && batch.complete
        % Reuse cached values when toggling; leave playback and the frame alone.
        setts.showConsistency = ~setts.showConsistency;
        initConsistencyControls();
        updateDraw();
        return;
    end
    setts.showConsistency = true;
    idx = data.current_idx;
    qList = getValidQueries();
    nQ = min(setts.N,size(data.INF.est_xy,3));
    batch = struct('sample',idx,'threshold',setts.T,'numQueries',nQ, ...
        'bootstrapCount',consistencyBootstrapCount(),'seed',setts.consistency.Seed, ...
        'sourcePath',data.INF.sourcePath,'imageWidth',data.INF.imageWidth, ...
        'queries',qList,'p_values',nan(1,nQ),'computed',false(1,nQ), ...
        'track_results',{cell(1,nQ)},'track_inputs',{cell(1,nQ)},'complete',false);
    if ~isfield(data,'consistency'), data.consistency = {}; end
    data.consistency{idx} = batch;
    oldString = btnConsistency.String;
    controls = findall(ctrlPanel,'Type','uicontrol');
    previousEnable = get(controls,'Enable');
    set(controls,'Enable','off');
    btnConsistency.String = 'Computing...';
    finish = onCleanup(@()restoreConsistencyControls(oldString,controls,previousEnable));
    stopPlayTimer();
    if isvalid(btnPlay), btnPlay.Value = 0; btnPlay.String = 'Play'; end
    updateDraw();
    drawnow;

    for t = 1:numel(qList)
        q = qList(t);
        if ~isgraphics(gfig), error('SPTnet:GenerativeCancelled','Visualization window was closed.'); end
        try
            input = getConsistencyInput(q);
            batch.track_inputs{q} = input;
            result = SPTnet_trajectory_compatibility(input.xy,input.H,input.D, ...
                'Frames',input.frames,'Seed',batch.seed,'BootstrapCount',batch.bootstrapCount, ...
                'ProgressCallback',@(progress)showConsistencyProgress(progress,q,t,numel(qList)));
        catch err
            if ~isgraphics(gfig) || strcmp(err.identifier,'SPTnet:GenerativeCancelled'), rethrow(err); end
            result = struct('supported',false,'p_compatibility',NaN,'status','error','reason',err.message);
        end
        batch.track_results{q} = result;
        batch.computed(q) = true;
        if result.supported, batch.p_values(q) = result.p_compatibility; end
        data.consistency{idx} = batch;
        updateDraw();
        drawnow;
    end
    if ~isgraphics(gfig), error('SPTnet:GenerativeCancelled','Visualization window was closed.'); end
    batch.complete = true;
    data.consistency{idx} = batch;
    setappdata(gfig,'SPTnetGenerativeLastResult',batch);
    setappdata(gfig,'SPTnetTrajectoryLastResult',batch);
    showConsistencySummary(batch);
    updateDraw();
end

function showConsistencyProgress(~,q,track,total)
    if ~isgraphics(gfig)
        error('SPTnet:GenerativeCancelled','Visualization window was closed.');
    end
    txtConsistencyStatus.String = sprintf('Computing p-value for track %d/%d (query %d)...',track,total,q);
    drawnow;
end

function showConsistencySummary(batch)
    if isempty(batch.queries)
        txtConsistencyStatus.String = 'No tracks pass the current threshold.';
    elseif ~setts.showConsistency
        txtConsistencyStatus.String = 'Compatibility p-values are hidden. Turn on to show the saved results.';
    else
        nValues = sum(isfinite(batch.p_values(batch.queries)));
        txtConsistencyStatus.String = sprintf('Sample %d: p-values shown for %d / %d tracks; %d N/A.', ...
            batch.sample,nValues,numel(batch.queries),numel(batch.queries)-nValues);
    end
end

function restoreConsistencyControls(label,controls,previousEnable)
    for ii = 1:numel(controls)
        if isgraphics(controls(ii)), controls(ii).Enable = previousEnable{ii}; end
    end
    if isgraphics(btnConsistency)
        btnConsistency.String = label;
        initConsistencyControls();
    end
end

function onChangeN(src,~)
    v = round(str2double(src.String));
    if ~isfinite(v) || v < 1
        src.String = num2str(setts.N);
        return;
    end
    setts.N = v;
    invalidateConsistency();
    initConsistencyControls();
    updateDraw();
end

function onChangeT(src,~)
    v = str2double(src.String);
    if ~isfinite(v) || v < 0 || v > 1
        src.String = num2str(setts.T);
        return;
    end
    setts.T = v;
    invalidateConsistency();
    initConsistencyControls();
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
    if isfield(data,'INF')
        n = size(data.INF.est_xy,1);
    elseif isfield(data,'GT')
        n = size(data.GT.timelapsedata,4);
    elseif isfield(data,'TIFF')
        n = 1;
    else
        return;
    end

    popData.Enable = 'on';
    popData.String = arrayfun(@num2str,1:n,'UniformOutput',false);
    popData.Value  = min(data.current_idx,n);
    data.current_idx = popData.Value;
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
    batch = getConsistencyResults();

    % choose background
    if isfield(data,'GT') && idx<=size(data.GT.timelapsedata,4) && fr<=size(data.GT.timelapsedata,3)
        Iraw = uint8(rescale(data.GT.timelapsedata(:,:,fr,idx))*255);
        Idisp = mat2gray(Iraw);
    elseif isfield(data,'TIFF') && fr<=size(data.TIFF,3)
        Iraw = data.TIFF(:,:,fr);
        % flip horizontally then rotate 90° 
        Iraw = fliplr(Iraw);
        Iraw = rot90(Iraw,1);
        Idisp = mat2gray(Iraw);
    elseif isfield(data,'INF')
        % The track labels also work when only an inference MAT file is loaded.
        Idisp = zeros(data.INF.imageWidth,data.INF.imageWidth);
    else
        hold(ax,'off');
        return;
    end

    [imgH, imgW] = size(Idisp);

    imshow(repmat(Idisp,[1,1,3]),'Parent',ax);
    axis(ax,'image','off');

    % ---------------- GT overlay (frame-by-frame + clip to FOV) ----------------
    if chkGT.Value && isfield(data,'GT') && isfield(data.GT,'traceposition') && ...
            idx<=size(data.GT.traceposition,1)
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
    if isfield(data,'INF') && idx<=size(data.INF.est_xy,1) && fr<=size(data.INF.est_xy,2)
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

                label = sprintf('H=%.2f,D=%.2f',data.INF.H(idx,q),data.INF.C(idx,q));
                text(ax,x,y-2,label,'Color',cmap(q,:),'FontSize',10, ...
                    'VerticalAlignment','middle','Tag','SPTnetTrackLabel', ...
                    'UserData',struct('sample',idx,'query',q));
                if setts.showConsistency && ~isempty(batch) && ismember(q,batch.queries)
                    if ~batch.computed(q)
                        pLabel = 'p=...';
                    elseif isfinite(batch.p_values(q))
                        pLabel = sprintf('p=%.4f',batch.p_values(q));
                    else
                        pLabel = 'p=N/A';
                    end
                    text(ax,x,y+2,pLabel,'Color',cmap(q,:),'FontSize',10, ...
                        'VerticalAlignment','top','Tag','SPTnetPValueLabel','UserData',q);
                end
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
