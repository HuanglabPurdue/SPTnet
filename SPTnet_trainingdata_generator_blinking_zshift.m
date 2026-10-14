% Main script for generating simulation videos for SPTnet training
% Version 2.3
% (C) Copyright 2025                The Huang Lab
%
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
%
%     Author: Cheng Bi, Jan 2026
%%
SPTnet_trainingdata_generator_GUI

function SPTnet_trainingdata_generator_GUI
    addpath(genpath(fullfile(pwd, 'PSF-toolbox'))) %addpath for PSF-toolbox
    if exist('newim', 'file') ~= 2, error('DIPimage is not available. Please make sure the required "DIPimage" is successfully installed and activated'); end % Make sure dipimage is available
    fig = uifigure('Name','SPTnet Training Video Generator','Position',[400 100 700 900]);  
    fig.UserData = struct( ...
        'ZMotion', defaultZMotionSettings(), ...
        'Blinking', defaultBlinkingSettings());
    handles.fig = fig;
    % panel 1
    psfNames = {'NA','Lambda','RefractiveIndex','OTFscale_SigmaX','OTFscale_SigmaY', ...
                'Pixelsize','PSFsize','nMed'};
    psfDefaults = [1.49, 0.69, 1.518, 0.95, 0.95, 0.157, 128, 1.33];
    [handles.psf, psfPanel] = createPanel(fig, 'PSF & Imaging setup', [10 470 280 420], psfNames, psfDefaults);
    
    % Panel is 280x420;
    % horizontal geometry
    xLabel = 10;      wLabel = 110;
    xMin   = 140;     wMin   = 50;
    xDash  =+ xMin + wMin + 5;
    xMax   = xDash + 10 + 5;
    wMax   = 50;

    % Vertical positions
    yPhoton = 90;
    yBg     = 60;
    yPerlin = 30;

    % Photon range: Photon [min] – [max]
    uilabel(psfPanel, ...
        'Position',[xLabel yPhoton wLabel 22], ...
        'Text','Photon', ...
        'Interpreter','none');
    handles.psf.Photon_min = uieditfield(psfPanel,'numeric', ...
        'Position',[xMin yPhoton wMin 22], ...
        'Value',300);           % default min photon
    uilabel(psfPanel, ...
        'Position',[xDash yPhoton 10 22], ...
        'Text','–', ...
        'HorizontalAlignment','center');
    handles.psf.Photon_max = uieditfield(psfPanel,'numeric', ...
        'Position',[xMax yPhoton wMax 22], ...
        'Value',5000);          % default max photon

    % Background range: Bg [min] – [max]
    uilabel(psfPanel, ...
        'Position',[xLabel yBg wLabel 22], ...
        'Text','Background (Bg)', ...
        'Interpreter','none');
    handles.psf.Bg_min = uieditfield(psfPanel,'numeric', ...
        'Position',[xMin yBg wMin 22], ...
        'Value',1);             % default min bg
    uilabel(psfPanel, ...
        'Position',[xDash yBg 10 22], ...
        'Text','–', ...
        'HorizontalAlignment','center');
    handles.psf.Bg_max = uieditfield(psfPanel,'numeric', ...
        'Position',[xMax yBg wMax 22], ...
        'Value',25);            % default max bg

    % Perlin background range: Perlin_Bg [min] – [max]
    uilabel(psfPanel, ...
        'Position',[xLabel yPerlin wLabel 22], ...
        'Text','Perlin Bg', ...
        'Interpreter','none');
    handles.psf.Perlin_Bg_min = uieditfield(psfPanel,'numeric', ...
        'Position',[xMin yPerlin wMin 22], ...
        'Value',0);             % default min perlin
    uilabel(psfPanel, ...
        'Position',[xDash yPerlin 10 22], ...
        'Text','–', ...
        'HorizontalAlignment','center');
    handles.psf.Perlin_Bg_max = uieditfield(psfPanel,'numeric', ...
        'Position',[xMax yPerlin wMax 22], ...
        'Value',10);            % default max perlin

    % panel 2
    simNames = {'Num_file', 'Videos_per_file', 'Frames', 'Image_dims'};
    simDefaults = [5, 100, 30, 64, 10];

    handles.sim = createPanel2col(fig, 'Video & Simulation Params', [10 50 280 250], simNames, simDefaults);
    handles.sim.MotionBlur = uicheckbox(fig, ...
    'Text', 'Add Motion Blur Effect', ...
    'Position', [20 80 150 20], ...
    'Value', false);    
    handles.zMotionButton = uibutton(fig, ...
        'Text', 'Z-axis Motion', ...
        'Position', [20 108 125 24], ...
        'Tooltip', 'Configure optional Z-axis motion', ...
        'ButtonPushedFcn', @(btn,event)openZMotionSettings(fig, btn));
    handles.blinkingButton = uibutton(fig, ...
        'Text', 'Blinking', ...
        'Position', [150 108 125 24], ...
        'Tooltip', 'Configure optional fluorophore blinking', ...
        'ButtonPushedFcn', @(btn,event)openBlinkingSettings(fig, btn));

    % panel for PSF preview
    handles.axPSF      = uiaxes(fig, 'Position',[290 660 200 200]);
    handles.axPSF.Title.String      = 'PSF (no noise)';
    initializeBlank(handles.axPSF, gray);
    
    handles.axPSFNoise = uiaxes(fig, 'Position',[490 660 200 200]);
    handles.axPSFNoise.Title.String = 'PSF (with noise)';
    initializeBlank(handles.axPSFNoise, gray);
    
    handles.axMag      = uiaxes(fig, 'Position',[290 450 200 200]);
    handles.axMag.Title.String      = 'Pupil Magnitude';
    initializeBlank(handles.axMag, jet);
    
    handles.axPhase    = uiaxes(fig, 'Position',[490 450 200 200]);
    handles.axPhase.Title.String    = 'Pupil Phase';
    initializeBlank(handles.axPhase, jet);
    
    % panel for Zernike
    zernPanel = uipanel(fig, ...
    'Title','Zernike Coefficients', ...
    'Position',[10 315 280 150]);

    % initial for [phase, magnitude]
    initialData = zeros(25,2);
    initialData(1,2) = 1; % default piston magnitude is 1 
    % create the table
    handles.zernTable = uitable(zernPanel, ...
        'Data',             initialData, ...
        'ColumnName',       {'Phase','Magnitude'}, ...
        'RowName',          arrayfun(@num2str,1:25,'Uni',0), ...
        'ColumnEditable',   [true true], ...
        'Position',         [0 30 280 95]);
    
    % create the box for video preview
    handles.axVideo = uiaxes(fig, ...
        'Position',[290 50 420 390]);
    handles.axVideo.Title.String      = 'Video Preview';
    axis(handles.axVideo,'off');  
    handles.axVideo.Color = [0 0 0];
    blank = zeros(64,64);
    imagesc(handles.axVideo, blank);
    axis(handles.axVideo,'image','off');
    colormap(handles.axVideo, gray);

    % Define Buttons
    uibutton(fig,'Text','Preview PSF',  'Position',[40 305 100 30], ... % +r-l,+u-d 
             'ButtonPushedFcn',@(btn,event)onpreview(handles,'preview'));
    uibutton(fig,'Text','Preview Video',  'Position',[40 40 100 30], ...
             'ButtonPushedFcn',@(btn,event)onpreviewvideo(handles,'preview'));     
    uibutton(fig,'Text','Generate', 'Position',[150 40 100 30],...
             'ButtonPushedFcn',@(btn,event)onGenerate(handles,'generate'));
    uibutton(fig, ...
            'Text','Reset Zernike', ...
            'Position',[150 305 100 30], ...   
            'ButtonPushedFcn',@(btn,event)resetZernike(handles));
end

function [h,p] = createPanel(parent,title,pos,names,defaults)
    p = uipanel(parent,'Title',title,'Position',pos);
    % leave ~50px for the title bar + top padding
    topMargin = 50;  
    y = pos(4) - topMargin;

    h = struct();
    for i = 1:numel(names)
        name = names{i};
        uilabel(p, ...
            'Position',[10 y 120 22], ...
            'Text',name, ...
            'Interpreter','none');
        fld = uieditfield(p,'numeric', ...
            'Position',[140 y 80 22], ...
            'Value',defaults(i));
        h.(name) = fld;
        y = y - 35;
    end
end

function h = createPanel2col(parent,title,pos,names,defaults)
    p = uipanel(parent,'Title',title,'Position',pos);

    topMargin = 45;
    rowStep   = 28;
    y0 = pos(4) - topMargin;

    % 2-column geometry
    xL_label = 5;   wLabel = 80;
    xL_fld   = 100;   wFld   = 40;
    xR_label = 145;
    xR_fld   = 235;

    n = numel(names);
    i = 1; row = 0;

    while i <= n
        y = y0 - row*rowStep;

        % left
        nameL = names{i};
        uilabel(p,'Position',[xL_label y wLabel 22],'Text',nameL,'Interpreter','none');
        h.(nameL) = uieditfield(p,'numeric','Position',[xL_fld y wFld 22],'Value',defaults(i));
        i = i + 1;

        % right
        if i <= n
            nameR = names{i};
            uilabel(p,'Position',[xR_label y wLabel+20 22],'Text',nameR,'Interpreter','none');
            h.(nameR) = uieditfield(p,'numeric','Position',[xR_fld y wFld 22],'Value',defaults(i));
            i = i + 1;
        end

        row = row + 1;
    end

    y_1 = y0 - row*rowStep;
    y_2 = y0 - (row+1)*rowStep;
    y_3 = y0 - (row+2)*rowStep;

    % Particle number range label + [min] - [max]
    uilabel(p,'Position',[xL_label y_1 90 22],'Text','Particle Number','Interpreter','none');
    h.p_num_min = uieditfield(p,'numeric','Position',[xL_fld y_1 40 22],'Value',1);
    uilabel(p,'Position',[xL_fld+42 y_1 10 22],'Text','–','HorizontalAlignment','center');
    h.p_num_max = uieditfield(p,'numeric','Position',[xL_fld+55 y_1 40 22],'Value',10);

    % Hurst range row: label + [min] - [max]
    uilabel(p,'Position',[xL_label y_2 90 22],'Text','Hurst Exponent','Interpreter','none');
    h.Hurst_min = uieditfield(p,'numeric','Position',[xL_fld y_2 40 22],'Value',0.0001);
    uilabel(p,'Position',[xL_fld+42 y_2 10 22],'Text','–','HorizontalAlignment','center');
    h.Hurst_max = uieditfield(p,'numeric','Position',[xL_fld+55 y_2 40 22],'Value',0.9999);

    % D range row on the right: label + [min] - [max]
    uilabel(p,'Position',[xL_label y_3 90 22],'Text','Diff. Coeff.','Interpreter','none');
    h.D_min = uieditfield(p,'numeric','Position',[xL_fld y_3 40 22],'Value',0.001);
    uilabel(p,'Position',[xL_fld+42 y_3 10 22],'Text','–','HorizontalAlignment','center');
    h.D_max = uieditfield(p,'numeric','Position',[xL_fld+55 y_3 40 22],'Value',0.5);
end


function onpreview(handles, action)
    % pull numeric values out into plain structs
    simParams = getParams(handles.sim);
    psfParams = getParams(handles.psf);
    
    zernData = handles.zernTable.Data; 
    zernikeCoefficients = zernData(:,1)';    % 1×25
    magnitudeCoefficients = zernData(:,2)';  % 1×25

    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag   = magnitudeCoefficients;

    imageSize = 128;                % Image size used for PSF generation
    
    % Define PSF parameters
    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag = magnitudeCoefficients;
    PRstruct.NA = psfParams.NA;           % Numerical aperture
    PRstruct.Lambda = psfParams.Lambda;          % Emission wavelength (microns)
    PRstruct.RefractiveIndex = psfParams.RefractiveIndex;     % Objective Immersion medium refractive index
    PRstruct.Pupil.phase = zeros(imageSize, imageSize);
    PRstruct.Pupil.mag = zeros(imageSize, imageSize);
    PRstruct.SigmaX = psfParams.OTFscale_SigmaX;       % Gaussian filter sigma (1/micron)
    PRstruct.SigmaY = psfParams.OTFscale_SigmaY;       % Gaussian filter sigma (1/micron)

    % Generate PSF object
    psfobj = PSF_zernike(PRstruct);
    psfobj.Boxsize = simParams.Image_dims;                     % Output PSF size
    psfobj.Pixelsize = psfParams.Pixelsize;                % Pixel size (microns)
    psfobj.PSFsize = psfParams.PSFsize;              % PSF image size
    psfobj.nMed = psfParams.nMed;                      % Sample medium refractive index
    
    Num = 5;                               % number of PSFs
    zpos = linspace(-2,2,Num)';             % z positions of the PSFs , unit is micron
    psfobj.Xpos = zeros(Num,1);             % x positions of the PSFs, unit is pixel
    psfobj.Ypos = zeros(Num,1);             % y positions of the PSFs, unit is pixel
    psfobj.Zpos = zpos;  
    % PSF computation
    psfobj.precomputeParam();
    psfobj.genPupil();
    psfobj.genPSF();
    psfobj.scalePSF('normal');  % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
    norm_parameter = sum(sum(psfobj.Pupil.mag));
    psf = psfobj.ScaledPSFs;    
    psf = psf./norm_parameter.*unifrnd(psfParams.Photon_min,psfParams.Photon_max);
    psf(isnan(psf)) = 0;
    % "noise" is a function in the DIPimage2.9 package
    noise_psf = single(noise(psf + unifrnd(psfParams.Bg_min,psfParams.Bg_max) + unifrnd(psfParams.Perlin_Bg_min,psfParams.Perlin_Bg_max) * perlin_noise(simParams.Image_dims), 'poisson'));
    M = size(psfobj.Pupil.mag,1);
    RC   = ceil(M/2);
    Rsub = floor(127/2);
    mag   = psfobj.Pupil.mag( RC-Rsub:RC+Rsub, RC-Rsub:RC+Rsub );
    phase = psfobj.Pupil.phase(RC-Rsub:RC+Rsub, RC-Rsub:RC+Rsub );
    % plot PSFs
    M   = size(psf,1);          
    cropSize = 15;
    startIdx = ceil((M-cropSize)/2)+1; 
    r = startIdx : startIdx+cropSize-1;
    imagesc(handles.axPSF, psf(r,r,ceil(size(psf,3)/2)))
    axis(handles.axPSF,'off')
    colormap(handles.axPSF, gray) 
    % noisy PSF
    imagesc(handles.axPSFNoise, noise_psf(r,r,ceil(size(psf,3)/2)))
    axis(handles.axPSFNoise,'off')
    colormap(handles.axPSFNoise, gray) 
    
    % pupil magnitude
    imagesc(handles.axMag, mag)
    axis(handles.axMag,'image','off')
    cm = [jet(256)]; 
    colormap(handles.axMag, cm)   % zero is black, high is white
    
    % pupil phase 
    imagesc(handles.axPhase, phase)
    axis(handles.axPhase,'image','off')
    cm = [jet(256)];          % first row = black for 0
    colormap(handles.axPhase, cm)

    switch action
      case 'preview'
%         dipshow(timelapsedata(:,:,:,1));
      case 'generate'
        msgbox('Simulation Completed!','Success','none');
    end
    beep;
end

function onpreviewvideo(handles, action)
    simParams = getParams(handles.sim);
    psfParams = getParams(handles.psf);
    zMotionParams = getZMotionSettings(handles.fig);
    blinkingParams = getBlinkingSettings(handles.fig);

    zernData = handles.zernTable.Data; 
    zernikeCoefficients = zernData(:,1)';    % 1×25
    magnitudeCoefficients = zernData(:,2)';  % 1×25

    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag   = magnitudeCoefficients;

    imageSize = 128;          
    % Define PSF parameters
    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag = magnitudeCoefficients;
    PRstruct.NA = psfParams.NA;           % Numerical aperture
    PRstruct.Lambda = psfParams.Lambda;          % Emission wavelength (microns)
    PRstruct.RefractiveIndex = psfParams.RefractiveIndex;     % Objective Immersion medium refractive index
    PRstruct.Pupil.phase = zeros(imageSize, imageSize);
    PRstruct.Pupil.mag = zeros(imageSize, imageSize);
    PRstruct.SigmaX = psfParams.OTFscale_SigmaX;       % Gaussian filter sigma (1/micron)
    PRstruct.SigmaY = psfParams.OTFscale_SigmaY;       % Gaussian filter sigma (1/micron)

    % Generate PSF object
    psfobj = PSF_zernike(PRstruct);
    psfobj.Boxsize = simParams.Image_dims;                     % Output PSF size
    psfobj.Pixelsize = psfParams.Pixelsize;                % Pixel size (microns)
    psfobj.PSFsize = psfParams.PSFsize;              % PSF image size
    psfobj.nMed = psfParams.nMed;                      % Sample medium refractive index
    
    enableMotionBlur = handles.sim.MotionBlur.Value;

    % ---- Dataset Generation ----
    for datasetIndex = 1:1
        % Background generation
        bgLevel = unifrnd(psfParams.Bg_min,psfParams.Bg_max);
        Perlin_bg = unifrnd(psfParams.Perlin_Bg_min,psfParams.Perlin_Bg_max);
        oversampling = 10; % parameter for motion blur, upsampling multiplier
        % Initialize PSF accumulation
        if enableMotionBlur
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames*oversampling));
        else
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
        end
        % Generate multiple trajectories per dataset
        numParticles = randi([simParams.p_num_min, simParams.p_num_max]); % Random number of particles per videos
        moleculeCounter = 0;

        T = simParams.Frames;
        Nmin = simParams.p_num_min;  
        Nmax = simParams.p_num_max;
        Lmin = 2;        
        Lmax = T;   
        tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax);
        
        for particleIndex = 1:numParticles
            start_t = tracks(particleIndex).t0;
            duration{datasetIndex,particleIndex} = tracks(particleIndex).t1 - tracks(particleIndex).t0 + 1;
            blinkFrame = generateBlinkTrace(blinkingParams, duration{datasetIndex,particleIndex});
            % Generate fBM trajectory
            hurstExponent = unifrnd(simParams.Hurst_min, simParams.Hurst_max);
            diffusionCoefficient = unifrnd(simParams.D_min, simParams.D_max);                       
            if enableMotionBlur
                osduration{datasetIndex,particleIndex} = duration{datasetIndex,particleIndex}*oversampling;
                osstart_t = 1+(start_t-1)*oversampling;
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration{datasetIndex,particleIndex}, diffusionCoefficient);
                FBMx = (1/oversampling)^(hurstExponent)*trajectoryX;
                FBMy = (1/oversampling)^(hurstExponent)*trajectoryY;
                photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max); 
                xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                overstep = simParams.Frames*oversampling;
                psfobj.Zpos = zeros(overstep,1);
                psfobj.Xpos = nan(overstep,1);
                psfobj.Ypos = nan(overstep,1);
                activeSteps = osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1);
                zTrajectory = generateZTrajectory(zMotionParams, hurstExponent, ...
                    osduration{datasetIndex,particleIndex}, diffusionCoefficient, ...
                    psfParams.Pixelsize, (1/oversampling)^hurstExponent);
                psfobj.Xpos(activeSteps) = FBMx + xOffset;   %FBMx;             % x positions of the PSFs, unit is pixel
                psfobj.Ypos(activeSteps) = FBMy + yOffset;   %FBMy;             % y positions of the PSFs, unit is pixel
                psfobj.Zpos(activeSteps) = zTrajectory;      % z positions of the PSFs, unit is micron
                traceposition{datasetIndex,particleIndex} = single([mean(reshape(psfobj.Xpos,[oversampling,simParams.Frames]))', mean(reshape(psfobj.Ypos,[oversampling,simParams.Frames]))']);
                psfobj.precomputeParam();               % generate parameters for Fourier space operation
                psfobj.genPupil();                      % generate pupil function 
                psfobj.genPSF();                        % generate PSFs
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                psf = psfobj.ScaledPSFs;    
                psf_blur = psf./norm_parameter.*(photons/oversampling);
                if blinkingParams.Enabled
                    blinkMaskOS = zeros(overstep,1,'like',psf_blur);
                    blinkMaskOS(activeSteps) = repelem(blinkFrame, oversampling);
                    psf_blur = psf_blur .* reshape(blinkMaskOS, 1, 1, []);
                end
                psf_blur(isnan(psf_blur)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf_blur(:,:,:);
                temp = reshape(psf_all, simParams.Image_dims, simParams.Image_dims, oversampling, simParams.Frames);
                blurpsf = squeeze(sum(temp, 3));
            else
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, duration{datasetIndex,particleIndex}, diffusionCoefficient);
                % Randomize initial position
                xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                % Assign positions to PSF object
                psfobj.Xpos = nan(simParams.Frames,1);
                psfobj.Ypos = nan(simParams.Frames,1);
                psfobj.Zpos = zeros(simParams.Frames,1);
                activeSteps = start_t:(start_t+duration{datasetIndex,particleIndex} -1);
                zTrajectory = generateZTrajectory(zMotionParams, hurstExponent, ...
                    duration{datasetIndex,particleIndex}, diffusionCoefficient, ...
                    psfParams.Pixelsize, 1);
                psfobj.Xpos(activeSteps) = trajectoryX + xOffset; % x positions of the PSFs, unit is pixel
                psfobj.Ypos(activeSteps) = trajectoryY + yOffset; % y positions of the PSFs, unit is pixel
                psfobj.Zpos(activeSteps) = zTrajectory;           % z positions of the PSFs, unit is micron
                % Assign photons to PSF
                photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max); 
                % Store particle trace positions
                traceposition{datasetIndex, particleIndex} = single([psfobj.Xpos, psfobj.Ypos]);
                % PSF computation
                psfobj.precomputeParam();
                psfobj.genPupil();
                psfobj.genPSF();
                psfobj.scalePSF('normal');           % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psf = psfobj.ScaledPSFs;    
                psf = psf./norm_parameter.*photons;
                if blinkingParams.Enabled
                    blinkMask = zeros(simParams.Frames,1,'like',psf);
                    blinkMask(activeSteps) = blinkFrame;
                    psf = psf .* reshape(blinkMask, 1, 1, []);
                end
                psf(isnan(psf)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf(:,:,:);
                % Accumulate PSF for the entire video
                % Store parameters
            end
        end
        % Add background and noise
        if enableMotionBlur
            timelapsedata(:,:,:) = single(noise(blurpsf+ bgLevel + Perlin_bg*perlin_noise(simParams.Image_dims),'poisson'));
        else
            timelapsedata(:,:,:) = single(noise(psf_all + bgLevel + Perlin_bg * perlin_noise(simParams.Image_dims), 'poisson'));
        end
    end
    
    colormap(handles.axVideo,'gray');
    for t = 1:size(timelapsedata,3)
        imagesc(handles.axVideo, timelapsedata(:,:,t));
        axis(handles.axVideo,'image','off')
        drawnow
        pause(0.05)
    end
end

function onGenerate(handles, action)
    folder = uigetdir(pwd, 'Select folder to save simulated videos');           
    if folder == 0
        return  % user cancelled
    end

    simParams = getParams(handles.sim);
    psfParams = getParams(handles.psf);
    zMotionParams = getZMotionSettings(handles.fig);
    blinkingParams = getBlinkingSettings(handles.fig);
    
    % Outer loop for number of simulation files
    total_files = simParams.Num_file; % define number of files generated for the simulation data, SPTnet is trained using 20,000 video per files, and 10 files in total
    numvideos = simParams.Videos_per_file;
    waitbarHandle = waitbar(0, 'Generating simulation videos, please wait...'); % Initialize progress bar

    zernData = handles.zernTable.Data; 
    zernikeCoefficients = zernData(:,1)';    % 1×25
    magnitudeCoefficients = zernData(:,2)';  % 1×25

    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag   = magnitudeCoefficients;

    imageSize = 128;          
    % Define PSF parameters
    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag = magnitudeCoefficients;
    PRstruct.NA = psfParams.NA;           % Numerical aperture
    PRstruct.Lambda = psfParams.Lambda;          % Emission wavelength (microns)
    PRstruct.RefractiveIndex = psfParams.RefractiveIndex;     % Objective Immersion medium refractive index
    PRstruct.Pupil.phase = zeros(imageSize, imageSize);
    PRstruct.Pupil.mag = zeros(imageSize, imageSize);
    PRstruct.SigmaX = psfParams.OTFscale_SigmaX;       % Gaussian filter sigma (1/micron)
    PRstruct.SigmaY = psfParams.OTFscale_SigmaY;       % Gaussian filter sigma (1/micron)

    % Generate PSF object
    psfobj = PSF_zernike(PRstruct);
    psfobj.Boxsize = simParams.Image_dims;                     % Output PSF size
    psfobj.Pixelsize = psfParams.Pixelsize;                % Pixel size (microns)
    psfobj.PSFsize = psfParams.PSFsize;              % PSF image size
    psfobj.nMed = psfParams.nMed;                      % Sample medium refractive index
    
    enableMotionBlur = handles.sim.MotionBlur.Value;

    for fileindex = 1:total_files
        % Data storage initialization
        timelapsedata = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames, numvideos));
        bglabel = zeros(1, numvideos, 'single');
        Perlinbglabel = zeros(1, numvideos, 'single');
        traceposition = cell(numvideos, simParams.p_num_max); 
        ztraceposition = cell(numvideos, simParams.p_num_max);
        latenttraceposition = cell(numvideos, simParams.p_num_max);
        latentztraceposition = cell(numvideos, simParams.p_num_max);
        blink_label = cell(numvideos, simParams.p_num_max);
        emissionfraction = cell(numvideos, simParams.p_num_max);
        photontrace = cell(numvideos, simParams.p_num_max);
        Hlabel = cell(numvideos, simParams.p_num_max);
        Clabel = cell(numvideos, simParams.p_num_max);
        photonlabel = cell(numvideos, simParams.p_num_max);
        moleculeid = cell(numvideos, simParams.p_num_max);
        duration = cell(numvideos,simParams.p_num_max);
        oversampling = 10;
        % ---- Dataset Generation ----
        for datasetIndex = 1:numvideos
            % Background generation
            bgLevel = unifrnd(psfParams.Bg_min, psfParams.Bg_max);
            if psfParams.Perlin_Bg_min<1
                Perlin_bg = 0;
            else
                Perlin_bg = unifrnd(psfParams.Perlin_Bg_min, psfParams.Perlin_Bg_max);
            end
                % Initialize PSF accumulation
            if enableMotionBlur
                psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames*oversampling));
                blurpsf = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
            else
                psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
            end
            % Generate multiple trajectories per dataset
            numParticles = randi([simParams.p_num_min, simParams.p_num_max]); % Random number of particles per videos
            moleculeCounter = 0;
            
            T = simParams.Frames;
            Nmin = simParams.p_num_min;  
            Nmax = simParams.p_num_max;
            Lmin = 2;        
            Lmax = T;   
            tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax);
            
            for particleIndex = 1:numParticles
                start_t = tracks(particleIndex).t0;
                duration{datasetIndex,particleIndex} = tracks(particleIndex).t1 - tracks(particleIndex).t0 + 1;
                activeFrames = start_t:(start_t+duration{datasetIndex,particleIndex}-1);
                blinkFrame = generateBlinkTrace(blinkingParams, duration{datasetIndex,particleIndex});
                blinkFull = nan(simParams.Frames,1,'single');
                blinkFull(activeFrames) = single(blinkFrame);
                 
                % Generate fBM trajectory
                hurstExponent = unifrnd(simParams.Hurst_min, simParams.Hurst_max);
                diffusionCoefficient = unifrnd(simParams.D_min, simParams.D_max);
                if enableMotionBlur
                    osduration{datasetIndex,particleIndex} = duration{datasetIndex,particleIndex}*oversampling;
                    osstart_t = 1+(start_t-1)*oversampling;
                    [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration{datasetIndex,particleIndex}, diffusionCoefficient);
                    FBMx = (1/oversampling)^(hurstExponent)*trajectoryX;
                    FBMy = (1/oversampling)^(hurstExponent)*trajectoryY;
                    photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max);
                    xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                    yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                    overstep = simParams.Frames*oversampling;
                    psfobj.Zpos = zeros(overstep,1);
                    psfobj.Xpos = nan(overstep,1);
                    psfobj.Ypos = nan(overstep,1);
                    activeSteps = osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1);
                    zTrajectory = generateZTrajectory(zMotionParams, hurstExponent, ...
                        osduration{datasetIndex,particleIndex}, diffusionCoefficient, ...
                        psfParams.Pixelsize, (1/oversampling)^hurstExponent);
                    psfobj.Xpos(activeSteps) = FBMx + xOffset;   %FBMx;             % x positions of the PSFs, unit is pixel
                    psfobj.Ypos(activeSteps) = FBMy + yOffset;   %FBMy;             % y positions of the PSFs, unit is pixel
                    psfobj.Zpos(activeSteps) = zTrajectory;      % z positions of the PSFs, unit is micron

                    latentXY = single([mean(reshape(psfobj.Xpos,[oversampling,simParams.Frames]))', mean(reshape(psfobj.Ypos,[oversampling,simParams.Frames]))']);
                    zTrace = nan(overstep,1);
                    zTrace(activeSteps) = zTrajectory;
                    latentZ = single(mean(reshape(zTrace,[oversampling,simParams.Frames]))');
                    observedXY = latentXY;
                    observedZ = latentZ;
                    observedXY(blinkFull ~= 1,:) = NaN;
                    observedZ(blinkFull ~= 1) = NaN;
                    traceposition{datasetIndex,particleIndex} = observedXY;
                    ztraceposition{datasetIndex,particleIndex} = observedZ;
                    latenttraceposition{datasetIndex,particleIndex} = latentXY;
                    latentztraceposition{datasetIndex,particleIndex} = latentZ;
                    blink_label{datasetIndex,particleIndex} = blinkFull;
                    emissionfraction{datasetIndex,particleIndex} = blinkFull;
                    photontraceCurrent = zeros(simParams.Frames,1,'single');
                    photontraceCurrent(activeFrames) = single(photons) .* single(blinkFrame);
                    photontrace{datasetIndex,particleIndex} = photontraceCurrent;
                    psfobj.precomputeParam();               % generate parameters for Fourier space operation
                    psfobj.genPupil();                      % generate pupil function 
                    psfobj.genPSF();                        % generate PSFs
                    norm_parameter = sum(sum(psfobj.Pupil.mag));
                    psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                    psf = psfobj.ScaledPSFs;    
        %             psf = psf./norm_parameter.*photons;
                    psf_blur = psf./norm_parameter.*(photons/oversampling);
                    if blinkingParams.Enabled
                        blinkMaskOS = zeros(overstep,1,'like',psf_blur);
                        blinkMaskOS(activeSteps) = repelem(blinkFrame, oversampling);
                        psf_blur = psf_blur .* reshape(blinkMaskOS, 1, 1, []);
                    end
                    psf_blur(isnan(psf_blur)) = 0;
                    psf_all(:,:,:) = psf_all(:,:,:) + psf_blur(:,:,:);

                    temp = reshape(psf_all, simParams.Image_dims, simParams.Image_dims, oversampling, simParams.Frames);
                    blurpsf = squeeze(sum(temp, 3));
                    Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
                    Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
                    photonlabel{datasetIndex, particleIndex} = single(photons);
                    moleculeCounter = moleculeCounter + 1;
                    moleculeid{datasetIndex, particleIndex} = moleculeCounter;
                else 
                    [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, duration{datasetIndex,particleIndex}, diffusionCoefficient);
                    % Randomize initial position
                    xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                    yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                    % Assign positions to PSF object
                    psfobj.Xpos = nan(simParams.Frames,1);
                    psfobj.Ypos = nan(simParams.Frames,1);
                    psfobj.Zpos = zeros(simParams.Frames,1);
                    activeSteps = start_t:(start_t+duration{datasetIndex,particleIndex} -1);
                    zTrajectory = generateZTrajectory(zMotionParams, hurstExponent, ...
                        duration{datasetIndex,particleIndex}, diffusionCoefficient, ...
                        psfParams.Pixelsize, 1);
                    psfobj.Xpos(activeSteps) = trajectoryX + xOffset; % x positions of the PSFs, unit is pixel
                    psfobj.Ypos(activeSteps) = trajectoryY + yOffset; % y positions of the PSFs, unit is pixel
                    psfobj.Zpos(activeSteps) = zTrajectory;           % z positions of the PSFs, unit is micron
                    % Assign photons to PSF
                    photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max); 

                    % Store particle trace positions
                    latentXY = single([psfobj.Xpos, psfobj.Ypos]);
                    zTrace = nan(simParams.Frames,1);
                    zTrace(activeSteps) = zTrajectory;
                    latentZ = single(zTrace);
                    observedXY = latentXY;
                    observedZ = latentZ;
                    observedXY(blinkFull ~= 1,:) = NaN;
                    observedZ(blinkFull ~= 1) = NaN;
                    traceposition{datasetIndex, particleIndex} = observedXY;
                    ztraceposition{datasetIndex, particleIndex} = observedZ;
                    latenttraceposition{datasetIndex,particleIndex} = latentXY;
                    latentztraceposition{datasetIndex,particleIndex} = latentZ;
                    blink_label{datasetIndex,particleIndex} = blinkFull;
                    emissionfraction{datasetIndex,particleIndex} = blinkFull;
                    photontraceCurrent = zeros(simParams.Frames,1,'single');
                    photontraceCurrent(activeFrames) = single(photons) .* single(blinkFrame);
                    photontrace{datasetIndex,particleIndex} = photontraceCurrent;

                    % PSF computation
                    psfobj.precomputeParam();
                    psfobj.genPupil();
                    psfobj.genPSF();
                    psfobj.scalePSF('normal');
                    norm_parameter = sum(sum(psfobj.Pupil.mag));
                    psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                    psf = psfobj.ScaledPSFs;    
                    psf = psf./norm_parameter.*photons;
                    if blinkingParams.Enabled
                        blinkMask = zeros(simParams.Frames,1,'like',psf);
                        blinkMask(activeSteps) = blinkFrame;
                        psf = psf .* reshape(blinkMask, 1, 1, []);
                    end
                    psf(isnan(psf)) = 0;
                    psf_all(:,:,:) = psf_all(:,:,:) + psf(:,:,:);
                    % Accumulate PSF for the entire video
                    % Store parameters
                    Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
                    Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
                    photonlabel{datasetIndex, particleIndex} = single(photons);
                    moleculeCounter = moleculeCounter + 1;
                    moleculeid{datasetIndex, particleIndex} = moleculeCounter;
                end
            end
            % Add background and noise
            bglabel(datasetIndex) = single(bgLevel);
            Perlinbglabel(datasetIndex) = single(Perlin_bg);
            if enableMotionBlur
                timelapsedata(:,:,:,datasetIndex) = single(noise(blurpsf(:,:,:)+ bgLevel + Perlin_bg * perlin_noise(simParams.Image_dims),'poisson'));
            else
                timelapsedata(:,:,:,datasetIndex) = single(noise(psf_all + bgLevel + Perlin_bg * perlin_noise(simParams.Image_dims), 'poisson'));
            end
        end
        % ---- Save Data ----
        saveFileName = [folder,'\trainingvideos_', num2str(fileindex)];
        ZmotionSettings = zMotionParams;
        BlinkingSettings = blinkingParams;
        save(saveFileName, 'timelapsedata', 'Hlabel', 'Clabel', 'photonlabel', ...
            'bglabel', 'traceposition', 'ztraceposition', 'latenttraceposition', ...
            'latentztraceposition', 'blink_label', 'emissionfraction', ...
            'photontrace', 'ZmotionSettings', 'BlinkingSettings', 'moleculeid', ...
            'Perlinbglabel', 'duration', '-v7.3');
        % Update progress bar
        waitbar(fileindex / total_files, waitbarHandle, sprintf('Generating file %d of %d...', fileindex, total_files));
    end
    beep
    close(waitbarHandle)
    msgbox('Simulation Completed!','Success','none');
end
    
function initializeBlank(ax, cmap)
    axis(ax,'off');                
    ax.Color = [1 1 1];            
    blank = zeros(16,16);          
    imagesc(ax, blank);
    axis(ax,'image','off');
    colormap(ax, cmap);
end

function resetZernike(handles)
    % rebuild the default [phase, magnitude] matrix
    defaultData = zeros(25,2);
    defaultData(1,2) = 1;      % piston = 1 by default
    handles.zernTable.Data = defaultData;
end

function S = getParams(h)
    % h is a struct of uieditfield handles
    S = struct();
    fn = fieldnames(h);
    for i = 1:numel(fn)
        S.(fn{i}) = h.(fn{i}).Value;
    end
end

function settings = defaultBlinkingSettings()
    settings = struct( ...
        'Enabled', false, ...
        'Model', 'transition', ...
        'TransitionMatrix', [0.10 0.90; 0.05 0.95], ...
        'MeanOnFrames', 20, ...
        'MeanOffFrames', 2, ...
        'IndependentOnProbability', 0.95, ...
        'LognormalMedianOnFrames', 20, ...
        'LognormalSigmaOn', 0.5, ...
        'LognormalMedianOffFrames', 2, ...
        'LognormalSigmaOff', 0.8, ...
        'InitialState', 'OFF');
end

function settings = getBlinkingSettings(mainFig)
    settings = defaultBlinkingSettings();
    figData = mainFig.UserData;
    if isstruct(figData) && isfield(figData, 'Blinking')
        settings = figData.Blinking;
    end
end

function openBlinkingSettings(mainFig, buttonHandle)
    settings = getBlinkingSettings(mainFig);
    mainPosition = mainFig.Position;
    dialogWidth = 600;
    dialogHeight = 530;
    dialogPosition = [mainPosition(1) + (mainPosition(3)-dialogWidth)/2, ...
                      mainPosition(2) + (mainPosition(4)-dialogHeight)/2, ...
                      dialogWidth, dialogHeight];

    dlg = uifigure( ...
        'Name', 'Blinking Settings', ...
        'Position', dialogPosition, ...
        'Resize', 'off', ...
        'WindowStyle', 'modal');

    fields.Enable = uicheckbox(dlg, ...
        'Text', 'Enable fluorophore blinking', ...
        'Position', [25 485 220 22], ...
        'Value', settings.Enabled);

    uilabel(dlg, 'Text', 'Blinking model', 'Position', [25 445 105 22]);
    fields.Model = uidropdown(dlg, ...
        'Items', {'Two-state transition matrix', ...
                  'Geometric ON/OFF dwell times', ...
                  'Independent frame dropout', ...
                  'Log-normal ON/OFF dwell times'}, ...
        'ItemsData', {'transition', 'geometric', 'independent', 'lognormal'}, ...
        'Position', [140 445 275 22], ...
        'Value', settings.Model);

    panelPosition = [25 180 550 245];
    panels.transition = uipanel(dlg, ...
        'Title', 'Two-state transition matrix', ...
        'Position', panelPosition);
    uilabel(panels.transition, ...
        'Text', 'Rows are the current state; columns are the state in the next frame.', ...
        'Position', [20 185 500 22]);
    fields.TransitionMatrix = uitable(panels.transition, ...
        'Data', settings.TransitionMatrix, ...
        'ColumnName', {'Next OFF', 'Next ON'}, ...
        'RowName', {'Current OFF', 'Current ON'}, ...
        'ColumnEditable', [true true], ...
        'Position', [20 45 505 135]);
    uilabel(panels.transition, ...
        'Text', 'Each row must contain probabilities from 0 to 1 and sum to 1.', ...
        'Position', [20 12 500 22]);

    panels.geometric = uipanel(dlg, ...
        'Title', 'Geometric dwell-time model', ...
        'Position', panelPosition);
    uilabel(panels.geometric, ...
        'Text', 'Mean ON duration (frames)', ...
        'Position', [25 155 180 22]);
    fields.MeanOnFrames = uieditfield(panels.geometric, 'numeric', ...
        'Position', [225 155 100 22], ...
        'Value', settings.MeanOnFrames);
    uilabel(panels.geometric, ...
        'Text', 'Mean OFF duration (frames)', ...
        'Position', [25 110 180 22]);
    fields.MeanOffFrames = uieditfield(panels.geometric, 'numeric', ...
        'Position', [225 110 100 22], ...
        'Value', settings.MeanOffFrames);
    uilabel(panels.geometric, ...
        'Text', {'Alternating ON and OFF runs are drawn in whole frames.'; ...
                 'The requested values are the exact discrete mean dwell times.'}, ...
        'Position', [25 35 485 50]);

    panels.independent = uipanel(dlg, ...
        'Title', 'Independent frame-dropout model', ...
        'Position', panelPosition);
    uilabel(panels.independent, ...
        'Text', 'Probability that each frame is ON', ...
        'Position', [25 150 220 22]);
    fields.IndependentOnProbability = uieditfield(panels.independent, 'numeric', ...
        'Position', [265 150 100 22], ...
        'Value', settings.IndependentOnProbability);
    uilabel(panels.independent, ...
        'Text', {'After the initial frame, every frame is drawn independently.'; ...
                 'This is useful for random missed-emission/dropout augmentation.'}, ...
        'Position', [25 70 490 50]);

    panels.lognormal = uipanel(dlg, ...
        'Title', 'Log-normal dwell-time model', ...
        'Position', panelPosition);
    uilabel(panels.lognormal, 'Text', 'Median ON (frames)', ...
        'Position', [20 165 145 22]);
    fields.LognormalMedianOnFrames = uieditfield(panels.lognormal, 'numeric', ...
        'Position', [170 165 80 22], ...
        'Value', settings.LognormalMedianOnFrames);
    uilabel(panels.lognormal, 'Text', 'Log sigma ON', ...
        'Position', [285 165 105 22]);
    fields.LognormalSigmaOn = uieditfield(panels.lognormal, 'numeric', ...
        'Position', [405 165 80 22], ...
        'Value', settings.LognormalSigmaOn);
    uilabel(panels.lognormal, 'Text', 'Median OFF (frames)', ...
        'Position', [20 115 145 22]);
    fields.LognormalMedianOffFrames = uieditfield(panels.lognormal, 'numeric', ...
        'Position', [170 115 80 22], ...
        'Value', settings.LognormalMedianOffFrames);
    uilabel(panels.lognormal, 'Text', 'Log sigma OFF', ...
        'Position', [285 115 105 22]);
    fields.LognormalSigmaOff = uieditfield(panels.lognormal, 'numeric', ...
        'Position', [405 115 80 22], ...
        'Value', settings.LognormalSigmaOff);
    uilabel(panels.lognormal, ...
        'Text', {'Larger log sigma produces a wider distribution of dwell times,'; ...
                 'including occasional long ON periods or dark gaps.'}, ...
        'Position', [20 35 500 50]);

    fields.Model.ValueChangedFcn = ...
        @(src,event)updateBlinkingModelPanels(src.Value, panels);
    updateBlinkingModelPanels(settings.Model, panels);

    uilabel(dlg, 'Text', 'Initial state', 'Position', [25 120 105 22]);
    fields.InitialState = uidropdown(dlg, ...
        'Items', {'ON', 'OFF', 'Stationary distribution'}, ...
        'ItemsData', {'ON', 'OFF', 'stationary'}, ...
        'Position', [140 120 180 22], ...
        'Value', settings.InitialState);
    fields.OriginalSettings = settings;

    uibutton(dlg, ...
        'Text', 'Cancel', ...
        'Position', [390 25 80 28], ...
        'ButtonPushedFcn', @(btn,event)delete(dlg));
    uibutton(dlg, ...
        'Text', 'Apply', ...
        'Position', [490 25 80 28], ...
        'ButtonPushedFcn', @(btn,event)applyBlinkingSettings( ...
            mainFig, buttonHandle, dlg, fields));
end

function updateBlinkingModelPanels(selectedModel, panels)
    panelNames = fieldnames(panels);
    for panelIndex = 1:numel(panelNames)
        panels.(panelNames{panelIndex}).Visible = 'off';
    end
    panels.(char(selectedModel)).Visible = 'on';
end

function applyBlinkingSettings(mainFig, buttonHandle, dlg, fields)
    selectedModel = char(fields.Model.Value);
    originalSettings = fields.OriginalSettings;

    transitionMatrix = fields.TransitionMatrix.Data;
    transitionValid = true;
    try
        if iscell(transitionMatrix)
            transitionMatrix = cell2mat(transitionMatrix);
        end
        transitionMatrix = double(transitionMatrix);
    catch
        transitionValid = false;
        transitionMatrix = originalSettings.TransitionMatrix;
    end
    transitionValid = transitionValid && isequal(size(transitionMatrix), [2 2]) && ...
        ~any(~isfinite(transitionMatrix(:))) && ...
        ~any(transitionMatrix(:) < 0) && ~any(transitionMatrix(:) > 1) && ...
        ~any(abs(sum(transitionMatrix,2)-1) > 1e-8);
    if strcmpi(selectedModel, 'transition') && ~transitionValid
        uialert(dlg, ...
            'The transition matrix must be 2-by-2, contain numeric probabilities from 0 to 1, and have rows that sum to 1.', ...
            'Invalid Transition Matrix');
        return
    elseif ~transitionValid
        transitionMatrix = originalSettings.TransitionMatrix;
    end

    meanOnFrames = fields.MeanOnFrames.Value;
    meanOffFrames = fields.MeanOffFrames.Value;
    geometricValid = all(isfinite([meanOnFrames meanOffFrames])) && ...
        all([meanOnFrames meanOffFrames] >= 1);
    if strcmpi(selectedModel, 'geometric') && ~geometricValid
        uialert(dlg, ...
            'Geometric mean ON and OFF dwell times must each be at least 1 frame.', ...
            'Invalid Mean Dwell Time');
        return
    elseif ~geometricValid
        meanOnFrames = originalSettings.MeanOnFrames;
        meanOffFrames = originalSettings.MeanOffFrames;
    end

    lognormalMedianOn = fields.LognormalMedianOnFrames.Value;
    lognormalMedianOff = fields.LognormalMedianOffFrames.Value;
    lognormalSigmaOn = fields.LognormalSigmaOn.Value;
    lognormalSigmaOff = fields.LognormalSigmaOff.Value;
    lognormalValid = all(isfinite([lognormalMedianOn lognormalMedianOff ...
        lognormalSigmaOn lognormalSigmaOff])) && ...
        all([lognormalMedianOn lognormalMedianOff] >= 1) && ...
        all([lognormalSigmaOn lognormalSigmaOff] >= 0);
    if strcmpi(selectedModel, 'lognormal') && ~lognormalValid
        uialert(dlg, ...
            'Log-normal medians must each be at least 1 frame, and log sigma values must be nonnegative.', ...
            'Invalid Log-normal Dwell Time');
        return
    elseif ~lognormalValid
        lognormalMedianOn = originalSettings.LognormalMedianOnFrames;
        lognormalMedianOff = originalSettings.LognormalMedianOffFrames;
        lognormalSigmaOn = originalSettings.LognormalSigmaOn;
        lognormalSigmaOff = originalSettings.LognormalSigmaOff;
    end

    onProbability = fields.IndependentOnProbability.Value;
    independentValid = isfinite(onProbability) && ...
        onProbability >= 0 && onProbability <= 1;
    if strcmpi(selectedModel, 'independent') && ~independentValid
        uialert(dlg, 'The independent ON probability must be between 0 and 1.', ...
            'Invalid Probability');
        return
    elseif ~independentValid
        onProbability = originalSettings.IndependentOnProbability;
    end

    initialState = char(fields.InitialState.Value);

    settings = struct( ...
        'Enabled', logical(fields.Enable.Value), ...
        'Model', selectedModel, ...
        'TransitionMatrix', transitionMatrix, ...
        'MeanOnFrames', meanOnFrames, ...
        'MeanOffFrames', meanOffFrames, ...
        'IndependentOnProbability', onProbability, ...
        'LognormalMedianOnFrames', lognormalMedianOn, ...
        'LognormalSigmaOn', lognormalSigmaOn, ...
        'LognormalMedianOffFrames', lognormalMedianOff, ...
        'LognormalSigmaOff', lognormalSigmaOff, ...
        'InitialState', initialState);

    figData = mainFig.UserData;
    figData.Blinking = settings;
    mainFig.UserData = figData;

    if settings.Enabled
        modelName = blinkingModelDisplayName(settings.Model);
        buttonHandle.Tooltip = sprintf('%s blinking enabled', modelName);
    else
        buttonHandle.Tooltip = 'Configure optional fluorophore blinking';
    end
    buttonHandle.Text = 'Blinking';
    delete(dlg)
end

function modelName = blinkingModelDisplayName(model)
    switch lower(model)
        case 'transition'
            modelName = 'Matrix';
        case {'geometric', 'exponential'}
            modelName = 'Dwell';
        case 'independent'
            modelName = 'IID';
        case 'lognormal'
            modelName = 'Broad';
        otherwise
            modelName = model;
    end
end

function blinkTrace = generateBlinkTrace(settings, steps)
    steps = round(steps);
    if steps <= 0
        blinkTrace = false(0,1);
        return
    end
    if ~settings.Enabled
        % Preserve the original random-number sequence when blinking is off.
        blinkTrace = true(steps,1);
        return
    end
    blinkTrace = simulateSingleBlinkTrace(settings, steps);
end

function blinkTrace = simulateSingleBlinkTrace(settings, steps)
    switch lower(settings.Model)
        case 'transition'
            blinkTrace = simulateTransitionBlink(settings, steps);
        case {'geometric', 'exponential'}
            blinkTrace = simulateDwellBlink(settings, steps, false);
        case 'independent'
            blinkTrace = simulateIndependentBlink(settings, steps);
        case 'lognormal'
            blinkTrace = simulateDwellBlink(settings, steps, true);
        otherwise
            error('SPTnet:InvalidBlinkingModel', ...
                'Unknown blinking model: %s', settings.Model);
    end
end

function blinkTrace = simulateTransitionBlink(settings, steps)
    blinkTrace = false(steps,1);
    blinkTrace(1) = chooseInitialBlinkState(settings);
    transitionMatrix = settings.TransitionMatrix;
    for frameIndex = 2:steps
        currentRow = 1 + double(blinkTrace(frameIndex-1));
        blinkTrace(frameIndex) = rand >= transitionMatrix(currentRow,1);
    end
end

function blinkTrace = simulateIndependentBlink(settings, steps)
    blinkTrace = false(steps,1);
    blinkTrace(1) = chooseInitialBlinkState(settings);
    if steps > 1
        blinkTrace(2:end) = rand(steps-1,1) < settings.IndependentOnProbability;
    end
end

function blinkTrace = simulateDwellBlink(settings, steps, useLognormal)
    blinkTrace = false(steps,1);
    currentState = chooseInitialBlinkState(settings);
    firstFrame = 1;
    while firstFrame <= steps
        if useLognormal
            if currentState
                medianFrames = settings.LognormalMedianOnFrames;
                logSigma = settings.LognormalSigmaOn;
            else
                medianFrames = settings.LognormalMedianOffFrames;
                logSigma = settings.LognormalSigmaOff;
            end
            dwellFrames = max(1, round(exp(log(medianFrames) + logSigma*randn)));
        else
            if currentState
                meanFrames = settings.MeanOnFrames;
            else
                meanFrames = settings.MeanOffFrames;
            end
            if meanFrames == 1
                dwellFrames = 1;
            else
                endProbability = 1/meanFrames;
                dwellFrames = floor(log(max(rand, realmin)) / ...
                    log(1-endProbability)) + 1;
            end
        end
        lastFrame = min(steps, firstFrame+dwellFrames-1);
        blinkTrace(firstFrame:lastFrame) = currentState;
        currentState = ~currentState;
        firstFrame = lastFrame + 1;
    end
end

function stateOn = chooseInitialBlinkState(settings)
    switch lower(settings.InitialState)
        case 'on'
            stateOn = true;
        case 'off'
            stateOn = false;
        case 'stationary'
            stateOn = rand < stationaryOnProbability(settings);
        otherwise
            error('SPTnet:InvalidBlinkInitialState', ...
                'Unknown initial blinking state: %s', settings.InitialState);
    end
end

function probabilityOn = stationaryOnProbability(settings)
    switch lower(settings.Model)
        case 'transition'
            pOffToOn = settings.TransitionMatrix(1,2);
            pOnToOff = settings.TransitionMatrix(2,1);
            transitionTotal = pOffToOn + pOnToOff;
            if transitionTotal > 0
                probabilityOn = pOffToOn / transitionTotal;
            else
                probabilityOn = 0.5;
            end
        case {'geometric', 'exponential'}
            probabilityOn = settings.MeanOnFrames / ...
                (settings.MeanOnFrames + settings.MeanOffFrames);
        case 'independent'
            probabilityOn = settings.IndependentOnProbability;
        case 'lognormal'
            expectedOn = settings.LognormalMedianOnFrames * ...
                exp(0.5*settings.LognormalSigmaOn^2);
            expectedOff = settings.LognormalMedianOffFrames * ...
                exp(0.5*settings.LognormalSigmaOff^2);
            probabilityOn = expectedOn / (expectedOn + expectedOff);
        otherwise
            probabilityOn = 0.5;
    end
    probabilityOn = min(1, max(0, probabilityOn));
end

function settings = defaultZMotionSettings()
    settings = struct( ...
        'Enabled', false, ...
        'ZMin', -1, ...
        'ZMax', 1, ...
        'Mode', 'fBM');
end

function settings = getZMotionSettings(mainFig)
    settings = defaultZMotionSettings();
    figData = mainFig.UserData;
    if isstruct(figData) && isfield(figData, 'ZMotion')
        settings = figData.ZMotion;
    end
end

function openZMotionSettings(mainFig, buttonHandle)
    settings = getZMotionSettings(mainFig);
    mainPosition = mainFig.Position;
    dialogWidth = 390;
    dialogHeight = 285;
    dialogPosition = [mainPosition(1) + (mainPosition(3)-dialogWidth)/2, ...
                      mainPosition(2) + (mainPosition(4)-dialogHeight)/2, ...
                      dialogWidth, dialogHeight];

    dlg = uifigure( ...
        'Name', 'Z-axis Motion Settings', ...
        'Position', dialogPosition, ...
        'Resize', 'off', ...
        'WindowStyle', 'modal');

    enableField = uicheckbox(dlg, ...
        'Text', 'Enable Z-axis movement', ...
        'Position', [25 235 180 22], ...
        'Value', settings.Enabled);

    uilabel(dlg, ...
        'Text', 'Z range from the in-focus plane (microns)', ...
        'Position', [25 195 260 22]);
    uilabel(dlg, 'Text', 'Minimum', 'Position', [25 160 70 22]);
    zMinField = uieditfield(dlg, 'numeric', ...
        'Position', [95 160 75 22], ...
        'Value', settings.ZMin);
    uilabel(dlg, 'Text', 'Maximum', 'Position', [195 160 70 22]);
    zMaxField = uieditfield(dlg, 'numeric', ...
        'Position', [265 160 75 22], ...
        'Value', settings.ZMax);

    uilabel(dlg, 'Text', 'Movement model', 'Position', [25 120 105 22]);
    modeField = uidropdown(dlg, ...
        'Items', {'fBM (same behavior as XY)', 'Uniform within Z range'}, ...
        'ItemsData', {'fBM', 'uniform'}, ...
        'Position', [135 120 205 22], ...
        'Value', settings.Mode);

    uilabel(dlg, ...
        'Text', {'fBM uses the particle''s XY Hurst exponent and diffusion'; ...
                 'coefficient. Uniform draws an independent Z value at'; ...
                 'each simulated time point.'}, ...
        'Position', [25 55 340 55]);

    uibutton(dlg, ...
        'Text', 'Cancel', ...
        'Position', [180 18 80 25], ...
        'ButtonPushedFcn', @(btn,event)delete(dlg));
    uibutton(dlg, ...
        'Text', 'Apply', ...
        'Position', [275 18 80 25], ...
        'ButtonPushedFcn', @(btn,event)applyZMotionSettings( ...
            mainFig, buttonHandle, dlg, enableField, zMinField, zMaxField, modeField));
end

function applyZMotionSettings(mainFig, buttonHandle, dlg, enableField, zMinField, zMaxField, modeField)
    zMin = zMinField.Value;
    zMax = zMaxField.Value;
    if ~isfinite(zMin) || ~isfinite(zMax)
        uialert(dlg, 'Z limits must be finite numeric values.', 'Invalid Z Range');
        return
    end
    if zMin > zMax
        uialert(dlg, 'The minimum Z value must not exceed the maximum Z value.', 'Invalid Z Range');
        return
    end

    settings = struct( ...
        'Enabled', logical(enableField.Value), ...
        'ZMin', zMin, ...
        'ZMax', zMax, ...
        'Mode', char(modeField.Value));
    figData = mainFig.UserData;
    figData.ZMotion = settings;
    mainFig.UserData = figData;

    if settings.Enabled
        buttonHandle.Tooltip = sprintf('%s Z-axis movement from %.4g to %.4g microns', ...
            settings.Mode, settings.ZMin, settings.ZMax);
    else
        buttonHandle.Tooltip = 'Configure optional Z-axis motion';
    end
    buttonHandle.Text = 'Z-axis Motion';
    delete(dlg)
end

function z = generateZTrajectory(settings, H, steps, C, pixelSize, samplingScale)
    if ~settings.Enabled
        z = zeros(steps,1);
        return
    end

    if settings.ZMin == settings.ZMax
        z = repmat(settings.ZMin, steps, 1);
        return
    end

    switch lower(settings.Mode)
        case 'fbm'
            zDisplacementPixels = fractional_brownian_motion_generator_1D(H, steps, C);
            zDisplacementPixels = zDisplacementPixels - zDisplacementPixels(1);
            zStart = unifrnd(settings.ZMin, settings.ZMax);
            z = zStart + samplingScale .* pixelSize .* zDisplacementPixels;
            z = reflectIntoRange(z, settings.ZMin, settings.ZMax);
        case 'uniform'
            z = unifrnd(settings.ZMin, settings.ZMax, steps, 1);
        otherwise
            error('SPTnet:InvalidZMotionMode', ...
                'Unknown Z movement model: %s', settings.Mode);
    end
end

function values = reflectIntoRange(values, lowerBound, upperBound)
    rangeWidth = upperBound - lowerBound;
    reflected = mod(values - lowerBound, 2*rangeWidth);
    beyondUpperBound = reflected > rangeWidth;
    reflected(beyondUpperBound) = 2*rangeWidth - reflected(beyondUpperBound);
    values = lowerBound + reflected;
end

function z = fractional_brownian_motion_generator_1D(H,steps,C)
    covariancematrix = zeros(steps,steps);
    for t = 1:steps
        for s = 1:steps
            covariancematrix(t,s) = 0.5*((t^(2*H)+s^(2*H)-abs(t-s)^(2*H)));
        end
    end
    L = chol(covariancematrix,'lower');
    Z = sqrt(2*C)*randn(steps,1);
    z = L*Z;
end


%% Function to generate fBM trace
function [x,y] = fractional_brownian_motion_generator_2D(H,steps,C)
    covariancematrix = zeros(steps,steps);
    for t = 1:steps
        for s = 1:steps
            covariancematrix(t,s) = 0.5*((t^(2*H)+s^(2*H)-abs(t-s)^(2*H)));
        end
    end
    % cholesky decomposition
    [L] = chol(covariancematrix,'lower');  %upper one is not unique
    X = sqrt(2*C)*randn(steps,1);
    Y = sqrt(2*C)*randn(steps,1);
    x = L*X;
    y = L*Y;
end

%% Function to generate uniform distributed duration time and presented molecule per frame
function tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax)
    assert(T >= 1);
    assert(numParticles >= 0);
    assert(Lmin >= 1 && Lmin <= Lmax);
    occ = zeros(1, T);
    tracks = struct('t0', cell(1,numParticles), 't1', cell(1,numParticles), 'L', cell(1,numParticles));

    % duration distribution uniform
    Ls = randi([Lmin Lmax], 1, numParticles);
    Ls(Ls > T) = T;

    % focusing on longer track first
    [Ls, order] = sort(Ls, 'descend');

    for k = 1:numParticles
        L = Ls(k);
        bestStarts = [];
        bestScore = inf;
        
        % valid tracks
        for t0 = 1:(T - L + 1)
            t1 = t0 + L - 1;
            occ2 = occ;
            occ2(t0:t1) = occ2(t0:t1) + 1;

            % score function  
            % score = max(occ2) - min(occ2);  
            score = sum((occ2 - mean(occ2)).^2);     %variance like   

            if score < bestScore - 1e-12
                bestScore = score;
                bestStarts = t0;
            elseif abs(score - bestScore) < 1e-12
                bestStarts(end+1) = t0;
            end
        end

        % introduce some randomness
        t0 = bestStarts(randi(numel(bestStarts)));
        t1 = t0 + L - 1;

        occ(t0:t1) = occ(t0:t1) + 1;

        tracks(k).t0 = t0;
        tracks(k).t1 = t1;
        tracks(k).L  = L;
    end

    % back to the original order
    tmp = tracks;
    tracks = struct('t0',{},'t1',{},'L',{});
    tracks(order) = tmp; 
end

