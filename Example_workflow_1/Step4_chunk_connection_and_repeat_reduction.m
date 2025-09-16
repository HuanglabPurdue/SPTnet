function stitch_connect_and_deduplicate
% Build connected trajectories across inferences and remove repeated detections
% Input:  stitched_results.mat  (must contain DetG from your previous script)
% Output: stitched_tracks.mat   (connected tracks, with H and D vectors)

%% -------------------- User parameters  --------------------
P.distConnect_px  = 1; % Max distance (px) to connect A(end) -> B(start)
P.maxGapFrames    = 2; % Max allowed frame gap between segments
P.minTrackLen     = 5; % Threshold used to drop short tracks (frames)
P.numOverlapSteps = 1; % The minimal number of overlapped step to be considered as a repeated-track
P.distRepeat_px   = 4; % Distance threshold (pixel) used to detemine the overlap steps
P.plot       = 1;  % 1 for quick plot at end, 0 without plot

%% -------------------- Load DetG --------------------
if exist('stitched_results.mat','file')
    S = load('stitched_results.mat','DetG');
else
    [f,p] = uigetfile('*.mat','Select stitched_results.mat (contains DetG)');
    assert(f~=0, 'No file selected.');
    S = load(fullfile(p,f),'DetG');
end
assert(isfield(S,'DetG'), 'DetG not found in the MAT file.');
DetG = S.DetG;

% DetG columns (Output from "stitch_and_display.m" ):
% 1=tg, 2=xg, 3=yg, 4=obj, 5=q, 6=x0, 7=y0, 8=t0, 9=D, 10=H, 11=inf_id
tg = DetG(:,1);  xg = DetG(:,2);  yg = DetG(:,3);
q  = DetG(:,5);  inf_id = DetG(:,11);
H  = DetG(:,10); D  = DetG(:,9);

%% -------------------- Build initial per-(inf_id,q) tracks --------------------
% Each track is a struct with fields: t,x,y,H,D, inf_id, q
pairs = unique(DetG(:,[11 5]), 'rows', 'stable'); % [inf_id, q]
tracks = cell(size(pairs,1),1);
for i = 1:size(pairs,1)
    mask = (inf_id==pairs(i,1) & q==pairs(i,2));
    Ti.t = tg(mask);
    Ti.x = xg(mask);
    Ti.y = yg(mask);
    Ti.H = H(mask);
    Ti.D = D(mask);
    Ti.inf_ids = repmat(pairs(i,1), sum(mask), 1);
    Ti.q_local = repmat(pairs(i,2), sum(mask), 1);
    % sort by time
    [Ti.t, ord] = sort(Ti.t);
    Ti.x = Ti.x(ord); Ti.y = Ti.y(ord);
    Ti.H = Ti.H(ord); Ti.D = Ti.D(ord);
    Ti.inf_ids = Ti.inf_ids(ord); Ti.q_local = Ti.q_local(ord);
    % filter tiny tracks up front
    if numel(Ti.t) >= P.minTrackLen
        tracks{i,1} = Ti;
    else
        tracks{i,1} = []; % discard tiny
    end
end
tracks = tracks(~cellfun('isempty',tracks));

%% -------------------- Connect across inferences --------------------
% Extra robustness params:
K_end   = 3;   % how many last points of A to consider
K_start = 3;   % how many first points of B to consider

% Greedy single-link from A -> best B where:
%   dt = tB_start - tA_end within [0 .. P.maxGapFrames]
%   min distance between tail(A, up to K_end) and head(B, up to K_start) <= P.distConnect_px
% Avoid multiple parents per B.

N = numel(tracks);
if N == 0
    merged = {};
    warning('No tracks to connect.'); 
else
    claimedAsNext = false(N,1);

    % Precompute starts & ends
    ends   = zeros(N,3);  % [t_end, x_end, y_end]
    starts = zeros(N,3);  % [t_start, x_start, y_start]
    for i=1:N
        ends(i,:)   = [tracks{i}.t(end),   tracks{i}.x(end),   tracks{i}.y(end)];
        starts(i,:) = [tracks{i}.t(1),     tracks{i}.x(1),     tracks{i}.y(1)];
    end

    % Build adjacency: next index for each A (0 if none)
    nextIdx = zeros(N,1);
    num_links = 0;

   for a = 1:N   % <-- IMPORTANT: colon, not multiply
        best = 0; bestCost = Inf;

        % Tail of A
        nA = numel(tracks{a}.t);
        idxA = max(1, nA-K_end+1) : nA;

        for b = 1:N
            if b==a || claimedAsNext(b), continue; end

            % Head of B
            nB = numel(tracks{b}.t);
            idxB = 1 : min(K_start, nB);

            % Evaluate all tail(A) vs head(B) pairs; keep best admissible match
            localBest = Inf; matched = false;
            for ia = idxA
                for ib = idxB
                    dt = tracks{b}.t(ib) - tracks{a}.t(ia);
                    if dt < 0 || dt > P.maxGapFrames, continue; end  % allow dt=0..maxGap
                    dx = tracks{b}.x(ib) - tracks{a}.x(ia);
                    dy = tracks{b}.y(ib) - tracks{a}.y(ia);
                    dist = hypot(dx,dy);
                    if dist <= P.distConnect_px
                        % Prefer smaller dt, then shorter distance
                        cost = dt + 0.001*dist;
                        if cost < localBest
                            localBest = cost;
                            matched = true;
                        end
                    end
                end
            end

            if matched && localBest < bestCost
                best = b; bestCost = localBest;
            end
        end

        if best > 0
            nextIdx(a) = best;
            claimedAsNext(best) = true;
            num_links = num_links + 1;
        end
    end

    % Merge chains A -> B -> C...
    visited = false(N,1);
    merged = {};
    for i = 1:N
        if visited(i), continue; end
        % follow chain forward
        chain = i;
        j = i;
        while nextIdx(j) ~= 0
            j = nextIdx(j);
            chain(end+1) = j; %#ok<AGROW>
        end
        % mark chain visited
        visited(chain) = true;

        % concatenate
        Tm.t = []; Tm.x = []; Tm.y = []; Tm.H = []; Tm.D = [];
        Tm.inf_ids = []; Tm.q_local = [];
        for k = 1:numel(chain)
            s = tracks{chain(k)};
            Tm.t = [Tm.t; s.t];
            Tm.x = [Tm.x; s.x];
            Tm.y = [Tm.y; s.y];
            Tm.H = [Tm.H; s.H];
            Tm.D = [Tm.D; s.D];
            Tm.inf_ids = [Tm.inf_ids; s.inf_ids];
            Tm.q_local = [Tm.q_local; s.q_local];
        end
        % ensure sorted by time after concat
        [Tm.t, ord] = sort(Tm.t);
        Tm.x = Tm.x(ord); Tm.y = Tm.y(ord);
        Tm.H = Tm.H(ord); Tm.D = Tm.D(ord);
        Tm.inf_ids = Tm.inf_ids(ord); Tm.q_local = Tm.q_local(ord);

        if numel(Tm.t) >= P.minTrackLen
            merged{end+1,1} = Tm; %#ok<AGROW>
        end
    end

    fprintf('[connector] links made: %d  | input tracks: %d  | merged tracks: %d\n', ...
            num_links, N, numel(merged));
end


%% -------------------- Remove repeated tracks (overlapping steps) --------------------
% Convert to cell arrays compatible with your API:
%   cellTracks: 1 x M cell, each {N_i x 3} [x y t]
%   keeped_H, keeped_C: 1 x M cell, each {N_i x 1} H/D vectors aligned to frames
M = numel(merged);
cellTracks = cell(1,M);
keeped_H   = cell(1,M);
keeped_C   = cell(1,M);
for m = 1:M
    cellTracks{m} = [merged{m}.x(:), merged{m}.y(:), merged{m}.t(:)];
    keeped_H{m} = merged{m}.H(:);
    keeped_C{m} = merged{m}.D(:);
end

% Wrap into a single "frame chunk" row to reuse your function signature
cellTracks_wrapped = cell(1, M);
H_wrapped = cell(1, M);
C_wrapped = cell(1, M);
for m = 1:M
    cellTracks_wrapped{m} = cellTracks{m};
    H_wrapped{m} = keeped_H{m};
    C_wrapped{m} = keeped_C{m};
end

[uniqueTracks_wrapped, num_repeat, H_keeped_f_wrapped, C_keeped_f_wrapped] = ...
    removeRepeatedTracks_byTime(cellTracks_wrapped, P.numOverlapSteps, P.distRepeat_px, H_wrapped, C_wrapped);

% Unwrap to flat lists
cellTracks_final = {};
H_final = {};
D_final = {};
row = uniqueTracks_wrapped; % 1 x K cell
for k = 1:numel(row)
    if isempty(row{k}), continue; end
    cellTracks_final{end+1} = row{k}; %#ok<AGROW>
    H_final{end+1} = H_keeped_f_wrapped{k}; %#ok<AGROW>
    D_final{end+1} = C_keeped_f_wrapped{k}; %#ok<AGROW>
end

%% -------------------- Save outputs --------------------
save('stitched_results_connected_remove_repeat.mat', ...
     'cellTracks_final', 'H_final', 'D_final', ...
     'P', 'num_repeat', '-v7.3');
fprintf('Saved: stitched_results_connected_remove_repeat.mat (connected & deduped tracks)\n');

%% -------------------- quick plot  --------------------
if P.plot && ~isempty(cellTracks_final)
    figure('Color','w','Name','Connected Tracks (quick view)','NumberTitle','off');
    cmap = lines(numel(cellTracks_final));
    hold on;
    for i = 1:numel(cellTracks_final)
        tr = cellTracks_final{i};
        plot(tr(:,1), tr(:,2), '-', 'LineWidth', 1.0, 'Color', cmap(1+mod(i-1,size(cmap,1)),:));
    end
    axis equal tight; set(gca,'YDir','reverse'); % image-like
    xlabel('x (px)'); ylabel('y (px)');
    title('Connected tracks after dedupe');
end
end

%% ------------------------------------------------------------
function [uniqueTracks,num_repeat,H_keeped_f,C_keeped_f] = removeRepeatedTracks_byTime(cellTracks, numOverlapSteps, distThreshold, H_keeped, C_keeped)
% cellTracks: 1 x M cell, each is [x y t] (t are integers)
% We consider two tracks "repeats" if they have >= numOverlapSteps frames in common
% (same t values) AND positions within distThreshold at those frames.

uniqueTracks = cell(size(cellTracks));
H_keeped_f = cell(size(H_keeped));
C_keeped_f = cell(size(C_keeped));
num_repeat = 0;

tracks = cellTracks(1,:);    % single row layout
H_temp  = H_keeped(1,:);
C_temp  = C_keeped(1,:);

uniqueRow = {};
unique_H  = {};
unique_C  = {};

for j = 1:length(tracks)
    if isempty(tracks{j}), continue; end
    isUnique = true;
    for k = 1:length(uniqueRow)
        if isempty(uniqueRow{k}), continue; end
        if isRepeatedTrack_byTime(tracks{j}, uniqueRow{k}, numOverlapSteps, distThreshold)
            isUnique = false;
            num_repeat = num_repeat + 1;
            break;
        end
    end
    if isUnique
        uniqueRow{end+1} = tracks{j}; 
        unique_H{end+1}  = H_temp{j};
        unique_C{end+1}  = C_temp{j};
    end
end

uniqueTracks(1,1:length(uniqueRow)) = uniqueRow;
H_keeped_f(1,1:length(unique_H)) = unique_H;
C_keeped_f(1,1:length(unique_C)) = unique_C;
end

function isRepeated = isRepeatedTrack_byTime(trackA, trackB, numOverlapSteps, distThreshold)
% trackA/trackB: [x y t]
% Count overlapping frames where distance <= threshold

tA = trackA(:,3); tB = trackB(:,3);
[commonT, ia, ib] = intersect(tA, tB, 'stable');
if isempty(commonT)
    isRepeated = false; return;
end

xyA = trackA(ia,1:2);
xyB = trackB(ib,1:2);
d   = hypot(xyA(:,1)-xyB(:,1), xyA(:,2)-xyB(:,2));
overlapCount = sum(d <= distThreshold);

isRepeated = overlapCount >= numOverlapSteps;
end
