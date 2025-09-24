%% load_multipage_tiffs_to_4D.mat
% Loads multi-page TIFF(s) into:
% timelapsedata : H x W x T x N  (Height, Width, Time, #Videos)
% Assumes all inputs are grayscale and share the same H, W.
clear; clc;

% --- 1) Select files ---
[fn, fp] = uigetfile({'*.tif;*.tiff','TIFF files (*.tif, *.tiff)'}, ...
                     'Select multi-page TIFF(s)', 'MultiSelect','on');
if isequal(fn,0), error('No files selected.'); end
if ischar(fn), fn = {fn}; end
N = numel(fn);

% --- 2) Inspect sizes and frame counts ---
H = []; W = []; maxT = 0; T_each = zeros(1,N);
for i = 1:N
    fpath = fullfile(fp, fn{i});
    info  = imfinfo(fpath);
    if isempty(info), error('File "%s" has no readable pages.', fn{i}); end
    hi = info(1).Height; wi = info(1).Width; Ti = numel(info);
    if isempty(H), H = hi; W = wi;
    else
        if hi~=H || wi~=W
            error('All files must have same HxW. "%s" is %dx%d, expected %dx%d.', ...
                   fn{i}, hi, wi, H, W);
        end
    end
    T_each(i) = Ti;
    maxT = max(maxT, Ti);
end

% --- 3) Preallocate output as single (zero-padded) ---
timelapsedata = zeros(H, W, maxT, N, 'single');

% --- 4) Load frames ---
fprintf('Loading %d file(s) → timelapsedata (H=%d, W=%d, T=%d, N=%d | single)\n', ...
        N, H, W, maxT, N);

for i = 1:N
    fpath = fullfile(fp, fn{i});
    info  = imfinfo(fpath);
    Ti    = numel(info);
    for t = 1:Ti
        frame = imread(fpath, t, 'Info', info);
        timelapsedata(:,:,t,i) = single(frame);  % <-- force single precision
    end
end

fprintf('Done. Size(timelapsedata) = [%d %d %d %d], class = %s\n', ...
        size(timelapsedata), class(timelapsedata));

% --- Optional: save ---
% [mf, mp] = uiputfile('timelapsedata.mat','Save variable as');
% if ~isequal(mf,0)
%     save(fullfile(mp,mf), 'timelapsedata','-v7.3'); % -v7.3 for big arrays
% end
