%% Compute CRLB matrix used for SPTnet training.
% Pre-allocate CRLB storage
CRLB_matrix_HD_frame = zeros(100,99,200,2,2);
warning('off','MATLAB:singularMatrix');
tic
for frames = 2:100 % for 2-100 frames
    % compute Fisher information for this # of steps (frames)
    I = information_noPoisson_fast(frames);

    % invert each 2×2 block to get CRLB
    for jj = 1:200 %Generalied diffusion coefficient ranging from 0.01 to 2.00 
        for ii = 1:99 % Hurst exponent ranging from 0.01 to 0.99
            CRLB_matrix_HD_frame(frames,ii,jj,:,:) = squeeze(I(ii,jj,:,:)) \ eye(2);
            % equivalent to inv(squeeze(I(ii,jj,:,:)))
        end
    end
end
toc
%% Visualize the matrix
colors = turbo(60);
% cbh.TickLabels = num2cell(1:2:30) ; 
for i = 5:1:60
[X, Y] = meshgrid(1:200, 1:99);
Z = squeeze(CRLB_matrix_HD_frame(i,:,:,1,1)); 
% surf(X, Y, Z, 'FaceColor', colors(i, :), 'EdgeColor', 'none');
surf(X, Y, Z, 'FaceColor', colors(i, :), 'EdgeColor', 'k', 'EdgeAlpha',0.2);
hold on
end
xlabel('Generalized diffusion coefficient');
ylabel('Hurst exponent');
zlabel('CRLB_H');

colormap(colors)
cbh = colorbar;
% cbh.Ticks = linspace(0, 1, 30) ;
% cbh.TickLabels = num2cell(2:2:30);
cbh.Ticks = [] ;
cbh.TickLabels = {};

xticks([10 20 30 40 50])
xticklabels({'0.1','0.2','0.3','0.4','0.5'})
yticks([0 10 20 30 40 50 60 70 80 90 100])
yticklabels({'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'})
set(gcf,'Position',[200 200 800 800])

%%
function I = information_noPoisson_fast(steps)
% Compute the Fisher information matrix I(ii,jj,:,:) for
% H = 0.01*ii, C = 0.01*jj, using a stable, vectorized Cholesky approach.

    % --- precompute grids and constants ---
    t      = (1:steps)';          % column vector
    s      = 1:steps;             % row vector
    [T,S]   = meshgrid(t,s);      
    D       = abs(T - S);         % |t-s|
    
    Hvals  = 0.01 * (1:99);       % 99 H‐values
    Cvals  = 0.01 * (1:200);       % 50 C‐values

    I = zeros(99,200,2,2);

    % mask to avoid log(0) on diagonal
    mask = (D > 0);

    for ii = 1:99
        H   = Hvals(ii);
        % precompute powers
        T2H = T .^ (2*H);
        S2H = S .^ (2*H);
        D2H = D .^ (2*H);
        
        % covariance kernel (∂R/∂C)
        B   = T2H + S2H - D2H;
        
        % build ∂/∂H factor without NaNs:
        L   = 2*log(T).*T2H + 2*log(S).*S2H;  % off-diagonal part
        L(mask) = L(mask) - 2 * D2H(mask) .* log(D(mask));
        % diagonal entries remain 2*log(t)*t^(2H) since D=0 ⇒ no subtraction

        for jj = 1:200
            C      = Cvals(jj);
            R      = C * B;            % covariance matrix
            diffC  =      B;           % ∂R/∂C
            diffH  = C  * L;           % ∂R/∂H

            % add tiny jitter to ensure SPD
            eps_jitter = 1e-8 * max(diag(R));
            R          = R + eps_jitter * eye(steps);

            % Cholesky‐based solve for X = R⁻¹ * diffH, W = R⁻¹ * diffC
            A    = chol(R,'lower');
            YH   = A \ diffH;    X = A' \ YH;
            YC   = A \ diffC;    W = A' \ YC;

            % Fisher information entries
            I11 = 0.5 * sum(sum( X .* X' ));
            I12 = 0.5 * sum(sum( X .* W' ));
            I22 = 0.5 * sum(sum( W .* W' ));

            I(ii,jj,1,1) = I11;
            I(ii,jj,1,2) = I12;
            I(ii,jj,2,1) = I12;
            I(ii,jj,2,2) = I22;
        end
    end
end
