function leading_eigenvalue_benchmark()
    % leading_eigenvalue_benchmark
    % Benchmarks the RieMARS_AdamW optimizer for finding the leading eigenvalue
    % on the sphere manifold using Manopt.
    %
    % This function sets up the leading eigenvalue problem, defines the
    % manifold, cost function, gradient, partial gradients, and runs the
    % RieMARS_AdamW optimizer. It then plots the optimality gap versus IFO
    % calls.
    %
    % Author: [Your Name]
    % Date: [Date]
    
    %% Initialization
    clc;
    close all;
    clear;
    
    % Add Manopt to MATLAB path (modify the path as per your setup)
    addpath('/my/directory/manopt/solvers');  % <-- Update this path
    
    % Set random seed for reproducibility
    rng('default');
    rng(0);
    
    %% Problem Parameters
    d = 500;       % Dimension of the sphere manifold
    r = 1;         % Number of leading eigenvalues (r=1 for leading eigenvalue)
    condition_number = 1e3;  % Condition number of matrix A
    N = d;         % Number of cost terms set to d to match dimensions
    
    %% Generate Data
    fprintf('Generating data ... ');
    
    % Define eigenvalues with a specified condition number
    lambdas = logspace(0, -log10(condition_number), d);  % 1-by-d vector
    
    % Generate a random orthogonal matrix Q using QR decomposition
    Q = randn(d, d);
    [Q, ~] = qr(Q, 0);  % Economy QR decomposition ensures Q is d-by-d orthogonal
    
    % Construct symmetric positive definite matrix A
    A = Q * diag(lambdas) * Q';
    
    % Decompose A into a sum of N rank-1 matrices for partial gradients
    % A = sum_{k=1}^N a_k a_k', where a_k = sqrt(lambdas(k)) * Q(:,k)
    % Since N = d, this decomposition is straightforward
    a_k = zeros(d, N);  % Initialize a_k as d-by-N matrix
    for k = 1:N
        a_k(:, k) = sqrt(lambdas(k)) * Q(:, k);  % Each a_k is d-by-1
    end
    
    % Store data points in a cell array for partial gradient computations
    data.x = mat2cell(a_k, d, ones(1, N));  % Each cell contains a d-by-1 vector
    
    fprintf('done.\n');
    
    %% Obtain True Solution
    fprintf('Computing true leading eigenvalue ... ');
    [v_true, lambda_true] = eigs(A, r, 'la');  % Leading eigenvalue and eigenvector
    v_true = v_true(:, 1);  % Extract the first eigenvector
    true_leading_eigenvalue = -0.5 * (v_true' * A * v_true);  % Compute the true leading eigenvalue
    fprintf('done.\n');
    fprintf('True leading eigenvalue: %.16e\n', true_leading_eigenvalue);
    
    %% Define Manifold
    problem.M = spherefactory(d);  % Sphere manifold S^{d-1}
    problem.ncostterms = N;        % Number of cost terms
    problem.data = data;           % Data structure
    
    %% Define Cost Function
    problem.cost = @cost;
    function f = cost(U)
        % cost: Compute the objective function value at U
        % U is a d x r matrix (r=1 for leading eigenvalue)
        f = 0;
        for k = 1:N
            a = data.x{k};
            f = f + (-0.5 * (U' * a)^2) / N;
        end
    end
    
    %% Define Euclidean Gradient
    problem.egrad = @egrad;
    function g = egrad(U)
        % egrad: Compute the Euclidean gradient at U
        g = zeros(d, r);
        for k = 1:N
            a = data.x{k};
            g = g + (- (a * a') * U) / N;
        end
    end
    
    %% Define Partial Euclidean Gradient
    problem.partialegrad = @partialegrad;
    function g = partialegrad(U, indices)
        % partialegrad: Compute the partial Euclidean gradient over a subset of indices
        % U is a d x r matrix
        % indices is a vector of indices for the mini-batch
        g = zeros(d, r);
        len = length(indices);
        for i = 1:len
            k = indices(i);
            a = data.x{k};
            g = g + (- (a * a') * U) / len;
        end
    end
    
    %% Plotting Parameters
    fs = 18;          % Font size
    fs_ax = 16;       % Axis font size
    lw = 3;           % Line width
    set(0, 'DefaultAxesFontName', 'ArialMT');
    
    %% Solver Options
    options = struct();
    options.batchsize = 32;        % Mini-batch size
    options.maxiter = 80;        % Maximum number of iterations
    options.checkperiod = 1;      % Frequency of checks
    options.verbosity = 1;         % Verbosity level
    options.lr = 0.01;             % Learning rate
    options.scheduler = 'cosine';  % Learning rate scheduler
    options.beta1 = 0.95;          % AdamW parameter beta1
    options.beta2 = 0.99;          % AdamW parameter beta2
    options.epsilon = 1e-12;       % AdamW parameter epsilon
    options.weight_decay = 0.0;    % Weight decay
    options.gamma = 0.025;         % Riemannian MARS parameter gamma
    
    %% Initialize Starting Point
    Uinit = problem.M.rand();  % Random point on the sphere manifold
    
    %% Run RieMARS_AdamW Optimizer
    fprintf('Running RieMARS_AdamW optimizer ...\n');
    [~, infos_marsadamw, ~] = RieMARS_AdamW(problem, Uinit, options);
    
    %% Compute Optimality Gap
    % Preallocate for speed
    outgap_marsadamw = zeros(length(infos_marsadamw), 1);
    
    for i = 1:length(infos_marsadamw)
        U = infos_marsadamw(i).x;  % Current point on the manifold
        f_current = 0;
        for k = 1:N
            a = data.x{k};
            f_current = f_current + (-0.5 * (U' * a)^2) / N;
        end
        outgap_marsadamw(i) = abs(f_current - true_leading_eigenvalue);
    end
    
    %% Plot Optimality Gap vs IFO Calls
    figure();
    semilogy([infos_marsadamw.gradcnt]/N, outgap_marsadamw, '-', 'LineWidth', lw); hold on;
    hold off;
    ax1 = gca;
    set(ax1, 'FontSize', fs_ax, 'FontWeight', 'bold');
    xlim([0 40]);  % Adjust based on actual IFO calls
    ylim([1e-9 1e4]);  % Adjust based on expected gaps
    yt = arrayfun(@(x) sprintf('10^{%d}', x), log10(get(gca, 'ytick')), 'UniformOutput', false); % Log scale labels
    set(gca, 'yticklabel', yt);
    xlabel('IFO/n', 'FontSize', fs, 'FontWeight', 'bold');
    ylabel('Optimality gap (log)', 'FontSize', fs, 'FontWeight', 'bold');
    legend('R-MARSAdamW', 'Location', 'best');
    title('Optimality Gap vs IFO Calls for R-MARSAdamW on Sphere Manifold', 'FontSize', fs, 'FontWeight', 'bold');
    
    %% Display Final Solution
    fprintf('Final optimality gap: %.16e\n', outgap_marsadamw(end));
    fprintf('True leading eigenvalue: %.16e\n', true_leading_eigenvalue);
end
