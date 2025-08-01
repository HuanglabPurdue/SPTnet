% Main script for generating simulation videos for SPTnet training
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
SPTnet_trainingdata_generator_GUI

function SPTnet_trainingdata_generator_GUI
    addpath(genpath(fullfile(pwd, 'PSF-toolbox'))) %addpath for PSF-toolbox
    if exist('newim', 'file') ~= 2, error('DIPimage is not available. Please make sure the required "DIPimage" is successfully installed and activated'); end % Make sure dipimage is available
    fig = uifigure('Name','SPTnet Training Video Generator','Position',[400 100 700 900]);  
    % panel 1
    psfNames = {'NA','Lambda','RefractiveIndex','OTFscale_SigmaX','OTFscale_SigmaY', ...
                'Pixelsize','PSFsize','nMed','Photon','Bg','Perlin_Bg'};
    psfDefaults = [1.49, 0.69, 1.518, 0.95, 0.95, 0.157, 128, 1.33, 1000, 25, 0];
    handles.psf = createPanel(fig, 'PSF & Imaging setup', [10 470 280 420], psfNames, psfDefaults);%+r-l,+u-d, width, height
    
    % panel 2
    simNames = {'Num_file', 'Videos_per_file', 'Frames', 'Image_dims', 'Max_particles'};
    simDefaults = [5, 100, 30, 64, 10];
    handles.sim = createPanel(fig, 'Video & Simulation Params', [10 50 280 250], simNames, simDefaults);
    handles.sim.MotionBlur = uicheckbox(fig, ...
        'Text', 'Add Motion Blur Effect', ...
        'Position', [20 80 150 20], ...
        'Value', false);    
    
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

function h = createPanel(parent,title,pos,names,defaults)
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
    psf = psf./norm_parameter.*psfParams.Photon;
    psf(isnan(psf)) = 0;
    % "noise" is a function in the DIPimage2.9 package
    noise_psf = single(noise(psf + psfParams.Bg + psfParams.Perlin_Bg * perlin_noise(simParams.Image_dims), 'poisson'));
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
        bgLevel = psfParams.Bg;
        Perlin_bg = psfParams.Perlin_Bg;
        oversampling = 10; % parameter for motion blur
        % Initialize PSF accumulation
        if enableMotionBlur
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames*oversampling));
        else
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
        end
        % Generate multiple trajectories per dataset
        numParticles = randi([1, simParams.Max_particles]); % Random number of particles per videos
        moleculeCounter = 0;
        for particleIndex = 1:numParticles
            % Generate fBM trajectory
            hurstExponent = unifrnd(0.0001, 0.9999);
            diffusionCoefficient = unifrnd(0.001, 0.5);
            duration{datasetIndex,particleIndex} = randi(simParams.Frames-1)+1; %range(2-30)
            start_t = randi(simParams.Frames+1-duration{datasetIndex,particleIndex});
            if enableMotionBlur
                osduration{datasetIndex,particleIndex} = duration{datasetIndex,particleIndex}*oversampling;
                osstart_t = 1+(start_t-1)*oversampling;
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration{datasetIndex,particleIndex}, diffusionCoefficient);
                FBMx = (1/oversampling)^(hurstExponent)*trajectoryX;
                FBMy = (1/oversampling)^(hurstExponent)*trajectoryY;
                photons = unifrnd(300,2000); 
                xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                overstep = simParams.Frames*oversampling;
                psfobj.Zpos = zeros(overstep,1);
                psfobj.Xpos = nan(overstep,1);
                psfobj.Ypos = nan(overstep,1);
                psfobj.Xpos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMx + xOffset;   %FBMx;             % x positions of the PSFs, unit is pixel
                psfobj.Ypos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMy + yOffset;   %FBMy;             % y positions of the PSFs, unit is pixel
                traceposition{datasetIndex,particleIndex} = single([mean(reshape(psfobj.Xpos,[oversampling,simParams.Frames]))', mean(reshape(psfobj.Ypos,[oversampling,simParams.Frames]))']);
                psfobj.precomputeParam();               % generate parameters for Fourier space operation
                psfobj.genPupil();                      % generate pupil function 
                psfobj.genPSF();                        % generate PSFs
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                psf = psfobj.ScaledPSFs;    
                psf_blur = psf./norm_parameter.*(psfParams.Photon/oversampling);
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
                psfobj.Zpos = zeros(simParams.Frames,1); % No z motion
                psfobj.Xpos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryX + xOffset; % x positions of the PSFs, unit is pixel
                psfobj.Ypos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryY + yOffset; % y positions of the PSFs, unit is pixel
                % Assign photons to PSF
                photons = unifrnd(300,2000); 
                % Store particle trace positions
                traceposition{datasetIndex, particleIndex} = single([psfobj.Xpos, psfobj.Ypos]);
                % PSF computation
                psfobj.precomputeParam();
                psfobj.genPupil();
                psfobj.genPSF();
                psfobj.scalePSF('normal');           % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psf = psfobj.ScaledPSFs;    
                psf = psf./norm_parameter.*psfParams.Photon;
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
        pause(0.05)          % adjust to taste
    end
end

function onGenerate(handles, action)
    folder = uigetdir(pwd, 'Select folder to save simulated videos');           
    if folder == 0
        return  % user cancelled
    end

    simParams = getParams(handles.sim);
    psfParams = getParams(handles.psf);
    
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
        traceposition = cell(numvideos, simParams.Max_particles); 
        Hlabel = cell(numvideos, simParams.Max_particles);
        Clabel = cell(numvideos, simParams.Max_particles);
        photonlabel = cell(numvideos, simParams.Max_particles);
        moleculeid = cell(numvideos, simParams.Max_particles);
        duration = cell(numvideos,simParams.Max_particles);
        oversampling = 10;
        % ---- Dataset Generation ----
        for datasetIndex = 1:numvideos
            % Background generation
            bgLevel = unifrnd(1, psfParams.Bg);
            if psfParams.Perlin_Bg<1
                Perlin_bg = 0;
            else
                Perlin_bg = unifrnd(1, psfParams.Perlin_Bg);
            end
                % Initialize PSF accumulation
            if enableMotionBlur
                psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames*oversampling));
            else
                psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
            end
            % Generate multiple trajectories per dataset
            numParticles = randi([1, simParams.Max_particles]); % Random number of particles per videos
            moleculeCounter = 0;
            for particleIndex = 1:numParticles
                % Generate fBM trajectory
                hurstExponent = unifrnd(0.0001, 0.9999);
                diffusionCoefficient = unifrnd(0.001, 0.5);
                duration{datasetIndex,particleIndex} = randi(simParams.Frames-1)+1; %range(2-30)
                start_t = randi(simParams.Frames+1-duration{datasetIndex,particleIndex});
                if enableMotionBlur
                    osduration{datasetIndex,particleIndex} = duration{datasetIndex,particleIndex}*oversampling;
                    osstart_t = 1+(start_t-1)*oversampling;
                    [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration{datasetIndex,particleIndex}, diffusionCoefficient);
                    FBMx = (1/oversampling)^(hurstExponent)*trajectoryX;
                    FBMy = (1/oversampling)^(hurstExponent)*trajectoryY;
                    photons = unifrnd(300,psfParams.Photon);
                    xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % make inital position not too close to the edge
                    yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); % 4 pixels away from the edge
                    overstep = simParams.Frames*oversampling;
                    psfobj.Zpos = zeros(overstep,1);
                    psfobj.Xpos = nan(overstep,1);
                    psfobj.Ypos = nan(overstep,1);
                    psfobj.Xpos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMx + xOffset;   %FBMx;             % x positions of the PSFs, unit is pixel
                    psfobj.Ypos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMy + yOffset;   %FBMy;             % y positions of the PSFs, unit is pixel

                    traceposition{datasetIndex,particleIndex} = single([mean(reshape(psfobj.Xpos,[oversampling,simParams.Frames]))', mean(reshape(psfobj.Ypos,[oversampling,simParams.Frames]))']);
                    psfobj.precomputeParam();               % generate parameters for Fourier space operation
                    psfobj.genPupil();                      % generate pupil function 
                    psfobj.genPSF();                        % generate PSFs
                    norm_parameter = sum(sum(psfobj.Pupil.mag));
                    psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                    psf = psfobj.ScaledPSFs;    
        %             psf = psf./norm_parameter.*photons;
                    psf_blur = psf./norm_parameter.*(photons/oversampling);
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
                    psfobj.Zpos = zeros(simParams.Frames,1); % No z motion
                    psfobj.Xpos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryX + xOffset; % x positions of the PSFs, unit is pixel
                    psfobj.Ypos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryY + yOffset; % y positions of the PSFs, unit is pixel
                    % Assign photons to PSF
                    photons = unifrnd(1,psfParams.Photon); 

                    % Store particle trace positions
                    traceposition{datasetIndex, particleIndex} = single([psfobj.Xpos, psfobj.Ypos]);

                    % PSF computation
                    psfobj.precomputeParam();
                    psfobj.genPupil();
                    psfobj.genPSF();
                    psfobj.scalePSF('normal');
                    norm_parameter = sum(sum(psfobj.Pupil.mag));
                    psfobj.scalePSF('normal');              % generate OTF scaled PSFs, 'IMM': PSFs with index mismatch aberration, 'normal': PSFs without index mismatch aberration 
                    psf = psfobj.ScaledPSFs;    
                    psf = psf./norm_parameter.*photons;
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
        save(saveFileName, 'timelapsedata', 'Hlabel', 'Clabel', 'photonlabel', 'bglabel', 'traceposition', 'moleculeid', 'Perlinbglabel','duration', '-v7.3');
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

