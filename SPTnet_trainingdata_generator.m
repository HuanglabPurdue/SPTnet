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
%% Generate Moving Particles Following Fractional Brownian Motion (fBM)
%%% Require DIPimage 2.8.1 %%%
%%% Add necessary path for PSF-toolbox %%%
addpath([pwd,'\PSF-toolbox'])
datasavepath = pwd;
% Outer loop for number of simulation files
total_files = 5; % define number of files generated for the simulation data, SPTnet is trained using 20,000 video per files, and 10 files in total
waitbarHandle = waitbar(0, 'Generating simulation videos, please wait...'); % Initialize progress bar

for fileindex = 1:total_files
    % ---- Setup Parameters ----
    % Image and optical parameters
    imageSize = 128;                % Image size used for PSF generation
    zernikeCoefficients = zeros(1, 25); % Zernike coefficients for pupil function
    zernikeCoefficients([5, 16, 19]) = [0, 0, 0]; % Add aberrations to pupil phase
    magnitudeCoefficients = zeros(1, 25);
    magnitudeCoefficients(1) = 1; % First order magnitude coefficient

    % Define PSF parameters
    PRstruct.Zernike_phase = zernikeCoefficients;
    PRstruct.Zernike_mag = magnitudeCoefficients;
    PRstruct.NA = 1.49;                      % Numerical aperture
    PRstruct.Lambda = 0.69;                  % Emission wavelength (microns)
    PRstruct.RefractiveIndex = 1.518;        % Immersion medium refractive index
    PRstruct.Pupil.phase = zeros(imageSize, imageSize);
    PRstruct.Pupil.mag = zeros(imageSize, imageSize);
    PRstruct.SigmaX = 0.95;                  % Gaussian filter sigma (1/micron)
    PRstruct.SigmaY = 0.95;                  % Gaussian filter sigma (1/micron)

    % Generate PSF object
    psfobj = PSF_zernike(PRstruct);
    psfobj.Boxsize = 64;                     % Output PSF size
    psfobj.Pixelsize = 0.157;                % Pixel size (microns)
    psfobj.PSFsize = imageSize;              % PSF image size
    psfobj.nMed = 1.33;                      % Sample medium refractive index

    % ---- Simulation Settings ----
    numFrames = 30;                          % Number of frames in simulation
    numvideos = 100;                         % Number of videos per files
    imageDims = 64;                          % Image dimensions (pixels)
    p_num_max = 10;                          % The maximum number of particles per video, query = p_num_max+10

    % Data storage initialization
    timelapsedata = single(zeros(imageDims, imageDims, numFrames, numvideos));
    bglabel = zeros(1, numvideos, 'single');
    Perlinbglabel = zeros(1, numvideos, 'single');
    traceposition = cell(numvideos, p_num_max); 
    Hlabel = cell(numvideos, p_num_max);
    Clabel = cell(numvideos, p_num_max);
    photonlabel = cell(numvideos, p_num_max);
    moleculeid = cell(numvideos, p_num_max);

    % ---- Dataset Generation ----
    for datasetIndex = 1:numvideos
        % Background generation
        bgLevel = unifrnd(1, 50);
        Perlin_bg = unifrnd(1, 50);

        % Initialize PSF accumulation
        psfAccumulated = single(zeros(imageDims, imageDims, numFrames));

        % Generate multiple trajectories per dataset
        numParticles = randi([1, p_num_max]); % Random number of particles per videos
        moleculeCounter = 0;

        for particleIndex = 1:numParticles
            % Generate fBM trajectory
            hurstExponent = unifrnd(0.0001, 0.9999);
            diffusionCoefficient = unifrnd(0.001, 0.5);
            [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, numFrames, diffusionCoefficient);

            % Randomize initial position
            xOffset = unifrnd(-(imageDims/2)+4, (imageDims/2)-4); % make inital position not too close to the edge
            yOffset = unifrnd(-(imageDims/2)+4, (imageDims/2)-4); % 4 pixels away from the edge

            % Assign positions to PSF object
            psfobj.Zpos = zeros(size(trajectoryX)); % No z motion
            psfobj.Xpos = trajectoryX + xOffset;
            psfobj.Ypos = trajectoryY + yOffset;
            
            % Assign photons to PSF
            photons = unifrnd(300,10000); 
            
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
            % Accumulate PSF for the entire video
            psfAccumulated = psfAccumulated + psf;

            % Store parameters
            Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
            Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
            photonlabel{datasetIndex, particleIndex} = single(photons);
            moleculeCounter = moleculeCounter + 1;
            moleculeid{datasetIndex, particleIndex} = moleculeCounter;
        end

        % Add background and noise
        timelapsedata(:,:,:,datasetIndex) = single(noise(psfAccumulated + bgLevel + Perlin_bg * perlin_noise(imageDims), 'poisson'));
        bglabel(datasetIndex) = single(bgLevel);
        Perlinbglabel(datasetIndex) = single(Perlin_bg);
    end
    % ---- Save Data ----
    saveFileName = [datasavepath,'\trainingvideos_', num2str(fileindex)];
    save(saveFileName, 'timelapsedata', 'Hlabel', 'Clabel', 'photonlabel', 'bglabel', 'traceposition', 'moleculeid', 'Perlinbglabel', '-v7.3');
    % Update progress bar
    waitbar(fileindex / total_files, waitbarHandle, sprintf('Generating file %d of %d...', fileindex, total_files));
end
close(waitbarHandle);

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
