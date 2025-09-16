function stitch_and_display
% Manual workflow:
%   1) Select all raw block files (block***_x#_y#_t#.mat; must contain variable named as "timelapsedata")
%   2) Enter stride (default values are block size (equivalent to having no overlap between blocks))
%   3) Select all result files from SPTnet inferences

% Outputs: "stitched_overlay.avi" in current folder.

%% --- 1) Select raw block files ---
[rawFiles, rawPath] = uigetfile('*.mat', 'Select ALL raw block*.mat files', 'MultiSelect','on');
assert(~isequal(rawFiles,0), 'No raw block files selected.');
if ischar(rawFiles), rawFiles = {rawFiles}; end

% Parse first raw to get block size
B0 = load(fullfile(rawPath, rawFiles{1}));
assert(isfield(B0,'timelapsedata'), 'Raw block file must contain variable "timelapsedata".');
blk = size(squeeze(B0.timelapsedata));   % [bx, by, bt]
if numel(blk)==2, blk(3)=1; end
bx = blk(1); by = blk(2); bt = blk(3);

% Parse block indices from filenames & find max indices
[ix_all, iy_all, it_all] = parse_block_indices(rawFiles);
nx = max(ix_all); ny = max(iy_all); nt = max(it_all);

%% --- 2) Enter stride ---
% Ask for stride (defaults to no overlap = block size)
defAns = {num2str(bx), num2str(by), num2str(bt)};
opts = struct('Resize','on', 'WindowStyle','normal', 'Interpreter','none');
dlg = inputdlg({'Stride X (pixels):','Stride Y (pixels):','Stride T (frames):'}, ...
                'stride', [1 25], defAns,opts);
assert(~isempty(dlg), 'Canceled.');
sx = str2double(dlg{1}); sy = str2double(dlg{2}); st = str2double(dlg{3});
assert(all([sx,sy,st] > 0), 'Stride must be positive.');

% Determine padded/full field size from grid & stride
Hp = (nx-1)*sx + bx;
Wp = (ny-1)*sy + by;
Tp = (nt-1)*st + bt;

% Reconstruct padded raw video
Vsum   = zeros(Hp,Wp,Tp,'single');
Vcount = zeros(Hp,Wp,Tp,'single');

for k = 1:numel(rawFiles)
    fn = rawFiles{k};
    tok = regexp(fn, 'block(\d+)_x(\d+)_y(\d+)_t(\d+)\.mat', 'tokens', 'once');
    if isempty(tok), warning('Skip (name pattern): %s', fn); continue; end
    ix = str2double(tok{2});
    iy = str2double(tok{3});
    it = str2double(tok{4});

    B = load(fullfile(rawPath, fn));
    if ~isfield(B,'timelapsedata'), warning('No timelapsedata in %s', fn); continue; end
    b = single(squeeze(B.timelapsedata));  % [bx,by,bt]
    if ndims(b)==2, b(:,:,end+1) = b; end

    x0 = (ix-1)*sx;  xr = (x0+1):(x0+bx);
    y0 = (iy-1)*sy;  yr = (y0+1):(y0+by);
    t0 = (it-1)*st;  tr = (t0+1):(t0+bt);

    Vsum(xr,yr,tr)   = Vsum(xr,yr,tr)   + b;
    Vcount(xr,yr,tr) = Vcount(xr,yr,tr) + 1;
end
mask = Vcount>0;
Vsum(mask) = Vsum(mask)./Vcount(mask);
V = Vsum;  % final padded volume HxWxT

%% --- 3) Select SPTnet output files (inference results from SPTnet) ---
[resFiles, resPath] = uigetfile('*.mat', 'Select ALL resultblock*.mat files', 'MultiSelect','on');
assert(~isequal(resFiles,0), 'No result files selected.');
if ischar(resFiles), resFiles = {resFiles}; end

Det = zeros(0,10);
query = 20; 
% Simple thresholds (match your style)
detect_thresh = 0.90;
min_track_len = 5;   % keep queries that cross threshold in >= X frames

for k = 1:numel(resFiles)
    rf = resFiles{k};
    tok = regexp(rf, 'resultblock(\d+)_x(\d+)_y(\d+)_t(\d+)\.mat', 'tokens', 'once');
    if isempty(tok), warning('Skip (name pattern): %s', rf); continue; end
    ix = str2double(tok{2});
    iy = str2double(tok{3});
    it = str2double(tok{4});

    load(fullfile(resPath, rf));
    estimation_xy_scale = estimation_xy*bx/2+bx/2;
    estimation_C = estimation_C*0.5; 
    % data format transfer for linux pc trained model
    xy = squeeze(permute(estimation_xy_scale,[1,3,2,4]));
    obj = squeeze(permute(obj_estimation,[1,4,3,2]));

    if isempty(xy) || isempty(obj), warning('No usable vars in %s', rf); continue; end

    Tloc = size(xy,1); 

    % Query filter by length in-threshold
    valid_q = false(1,query);
    for q=1:query
        valid_q(q) = sum(obj(:,q) >= detect_thresh) >= min_track_len;
    end

    % Shift local → global (1-based for plotting)
    x0 = (ix-1)*64; y0 = (iy-1)*64; t0 = (it-1)*st;
    for tloc = 1:Tloc
        tg = t0 + tloc;
        if tg<1 || tg>size(V,3), continue; end
        qs = find(valid_q & obj(tloc,:) >= detect_thresh);
        for q = qs
            xy_local = squeeze(xy(tloc,q,:)); % [2] = [x,y]
            xg = y0 + xy_local(1)+1;
            yg = x0 + xy_local(2)+1;
            Det(end+1,:) = [tg, xg, yg, obj(tloc,q), q,x0,y0,t0, estimation_C(q), estimation_H(q)]; %#ok<AGROW>
        end
    end
end

% Clip to field
Det(:,1) = max(1, min(Det(:,1), size(V,3)));
Det(:,2) = max(1, min(Det(:,2), size(V,2)));
Det(:,3) = max(1, min(Det(:,3), size(V,1)));

%% --- 4) Stitch results and overlay with original video ---
% DetG columns:
% 1=tg (frame, local), 2=xg (local x), 3=yg (local y), 4=obj (detection probablity), % 5=q (query id, local),
% 6=x0 (x shift), 7=y0 (y shift), 8=t0 (time shift), 9=D (generalized diffusion coefficient), 10=H (Hurst), 11=inf_id
% This code stitches to global coords and treats queries in each inf_id as independent tracks. 
% --- color coding ----------
colorMode = 'hurst'; % mode including 'hurst', 'diffusion', default
seedKey   = Det(:,6:8);                  % [x0 y0 t0]
[~,~,gid] = unique(seedKey, 'rows', 'stable');
DetG      = [Det, gid];                  % 11th col = inference/seed id
DetG      = sortrows(DetG, [11 1]);      % sort by seed, then time
save('stitched_results.mat','DetG','-v7.3');
fprintf('Saved: stitched_results.mat (DetG)\n');
% Build global table (keep original columns; ADD H,D for color choices)
tg_global = DetG(:,1);
x_global  = DetG(:,2);
y_global  = DetG(:,3);
p_det     = DetG(:,4);
q_local   = DetG(:,5);
inf_id    = DetG(:,11);
D_val     = DetG(:,9);
H_val     = DetG(:,10);

DG = [tg_global, x_global, y_global, p_det, q_local, inf_id, H_val, D_val];  % [1..8]
DG = sortrows(DG, [6 5 1]);   % sort by (inf_id, q_local, time)

% Precompute per-(inf_id,q) pairs (track identity)
pairs = unique(DG(:,[6 5]), 'rows', 'stable');   % [inf_id, q_local]
num_pairs = size(pairs,1);

% Defaults for continuous color maps
Ngrad = 256;
cmapGrad = jet(Ngrad);   % continuous map for H/D
cmapTrack = hsv(max(num_pairs,1));  % original wheel for track ID

% For continuous modes, compute one scalar per pair
pair_vals = nan(num_pairs,1);
dmin = 0; dmax = 0.5;   % default D range

switch colorMode
    case 'hurst'
        % Use H in [0,1] per pair (median to be robust; all rows share same H anyway)
        for i = 1:num_pairs
            mask = (DG(:,6)==pairs(i,1) & DG(:,5)==pairs(i,2));
            pair_vals(i) = median(DG(mask,7));  % H_val
        end
        % clamp to [0,1]
        pair_vals = min(max(pair_vals,0),1);

    case 'diffusion'
        % Ask for D range
        ansD = inputdlg({'D min:','D max:'}, 'D range', [1 25], {'0','0.5'}, opts);
        assert(~isempty(ansD), 'Canceled.');
        dmin = str2double(ansD{1});
        dmax = str2double(ansD{2});
        if ~(isfinite(dmin) && isfinite(dmax) && dmax>dmin)
            warning('Invalid D range; falling back to [0,0.5].');
            dmin = 0; dmax = 0.5;
        end
        for i = 1:num_pairs
            mask = (DG(:,6)==pairs(i,1) & DG(:,5)==pairs(i,2));
            pair_vals(i) = median(DG(mask,8));  % D_val
        end
        % normalize to [0,1] using user range
        pair_vals = (pair_vals - dmin) / (dmax - dmin);
        pair_vals = min(max(pair_vals,0),1);  % clamp

    otherwise
        % By Track (default): keep original behavior (categorical colors)
        % nothing to precompute
end
% -------------------------------------------------------

tailT        = Inf;         % tail length in frames (Inf for full history)
drawMarkers  = true;        % draw current detections as markers
breakOnGaps  = true;        % avoid lines across missing frames

% Writer / Figure
vout = VideoWriter('stitched_overlay.avi'); vout.FrameRate = 5; open(vout);
hf = figure('Color','w','Name','Stitched Overlay','NumberTitle','off');

T = size(V,3);   % total frames
for t = 1:T
    % background image
    frm = V(:,:,t);
    mn = min(frm,[],'all'); mx = max(frm,[],'all');
    if mx > mn, frm8 = uint8(255*((frm-mn)/(mx-mn))); else, frm8 = uint8(frm); end
    rgb = im2double(cat(3,frm8,frm8,frm8));
    imshow(rgb, 'InitialMagnification', 2000); hold on;
    set(gca,'FontSize',14,'FontName','Arial','FontWeight','bold');

    % time window [t0, t]
    if isfinite(tailT)
        t0 = max(1, t - tailT + 1);
    else
        t0 = 1;
    end
    win = (DG(:,1) >= t0) & (DG(:,1) <= t);
    if any(win)
        Dwin = DG(win,:);
        % map each row to its pair index
        [~, pair_ix] = ismember(Dwin(:,[6 5]), pairs, 'rows');

        % iterate over each (inf_id, q) track present in the window
        [uPairs, ~, uIdx] = unique(pair_ix, 'stable');
        for kk = 1:numel(uPairs)
            mask = (uIdx == kk);
            seg  = Dwin(mask, :);                  % [tg, x, y, p, q, inf, H, D]
            cix  = uPairs(kk);

            % choose color based on mode
            switch colorMode
                case 'hurst'
                    val = pair_vals(cix);                     % 0..1
                    col = cmapGrad( max(1, min(Ngrad, 1+floor(val*(Ngrad-1))) ), : );
                case 'diffusion'
                    val = pair_vals(cix);                     % 0..1 after normalization
                    col = cmapGrad( max(1, min(Ngrad, 1+floor(val*(Ngrad-1))) ), : );
                otherwise
                    % By Track (default)
                    col = cmapTrack( (mod(cix-1, size(cmapTrack,1)) + 1), : );
            end

            % sort by time within this track
            [~, ord] = sort(seg(:,1)); seg = seg(ord,:);

            % draw with optional breaks on missing frames
            if breakOnGaps
                f = seg(:,1);
                brk = find(diff(f) > 1);
                s = 1;
                for b = [brk(:); numel(f)]
                    plot(seg(s:b,2), seg(s:b,3), '-', 'LineWidth', 1.2, 'Color', col);
                    s = b + 1;
                end
            else
                plot(seg(:,2), seg(:,3), '-', 'LineWidth', 1.2, 'Color', col);
            end

            % draw current-frame marker if this track has a point at time t
            if drawMarkers
                cur = seg(seg(:,1) == t, :);
                if ~isempty(cur)
                    plot(cur(:,2), cur(:,3), 'x', 'MarkerSize', 6, 'LineWidth', 1.2, 'Color', col);
                end
            end
        end
    end

    drawnow;
    writeVideo(vout, getframe(gca));
    hold off
end
close(vout);
close(hf);
fprintf('Saved: stitched_overlay.avi\n');
end


function [ix_all, iy_all, it_all] = parse_block_indices(files)
ix_all = zeros(numel(files),1);
iy_all = zeros(numel(files),1);
it_all = zeros(numel(files),1);
for i=1:numel(files)
    tok = regexp(files{i}, 'block(\d+)_x(\d+)_y(\d+)_t(\d+)\.mat', 'tokens', 'once');
    assert(~isempty(tok), 'Filename pattern must be block***_x#_y#_t#.mat: %s', files{i});
    ix_all(i) = str2double(tok{2});
    iy_all(i) = str2double(tok{3});
    it_all(i) = str2double(tok{4});
end
end