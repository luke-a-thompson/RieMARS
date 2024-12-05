function test_grass_nleig()
% This example is motivated in the paper and is included in ManOpt
% "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
% Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
% SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
    
    clear 
    clc
    rng('default');
    rng(22);

    if ~exist('L', 'var') || isempty(L)
        n = 100;
        L = gallery('tridiag', n, -1, 2, -1);
    end
    
    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');
    
    if ~exist('k', 'var') || isempty(k) || k > n
        k = 5;
    end
    
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;
    end
    
    M = grassmannfactory(n, k);
    problem.M = M;
    
    % Cost function evaluation
    problem.cost =  @cost;
    function val = cost(X)
        rhoX = sum(X.^2, 2); % diag(X*X'); 
        val = 0.5*trace(X'*(L*X)) + (alpha/4)*(rhoX'*(L\rhoX));
    end
    
    % Euclidean gradient evaluation
    problem.egrad = @egrad;
    function g = egrad(X)
        rhoX = sum(X.^2, 2); % diag(X*X');
        g = L*X + alpha*diag(L\rhoX)*X;
    end
    
    x0 = M.rand();


    maxiter = 500;
    stepsize = 0.1;
    lambda = 1e-8;
    tolgradnorm = 1e-6;
    
    
    % compute optimal solution using rlbfgs
    options.tolgradnorm = 1e-10;
    options.memory = 5;
    [Xopt, costopt] = rlbfgs(problem, [], options);
    

    %% rgd with fixed stepsize
    % RieMARS_AdamW
    clear options
    options.batchsize = 100;
    options.maxiter = 1000;
    options.checkperiod = 50;
    options.verbosity = 50;
    options.lr = 0.01;
    options.scheduler = 'fixed';
    options.beta1 = 0.95;
    options.beta2 = 0.999;
    options.epsilon = 1e-12;
    options.weight_decay = 0.0;
    options.gamma = 0.025;
    % options.transport = 'ret_vector';
    [~, infos_marsadamw, ~] = RieMARS_AdamW(problem, x0, options);
    
    
    %% plots
    lw = 2.0;
    ms = 2.4;
    fs = 21;
    colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
          [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
          [152, 78, 163]/255, [77, 175, 74]/255}; 


    optgap_marsadamw = abs([infos_marsadamw.cost] - costopt);


    h1 = figure;
    semilogy([infos_marsadamw.iter], [infos_marsadamw.gradnorm], '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    xlabel('Iterations', 'fontsize', fs);
    ylabel('Gradnorm', 'fontsize', fs);
    legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);



    h2 = figure;
    semilogy([infos_marsadamw.iter], optgap_marsadamw, '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    xlabel('Iterations', 'fontsize', fs);
    ylabel('Optimality gap', 'fontsize', fs);
    legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);
    
    h3 = figure;
    semilogy([info_gd.time], optgap_gd, '-o', 'color', colors{1}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_ragd.time], optgap_ragd, '-^', 'color', colors{3},  'LineWidth', lw, 'MarkerSize',ms); hold on;
    semilogy([info_rnag.time], optgap_rnag, '-x', 'color', colors{7},  'LineWidth', lw, 'MarkerSize',ms); hold on;
    semilogy([info_rnagsc.time], optgap_rnag_sc, '-d', 'color', colors{6},  'LineWidth', lw, 'MarkerSize',ms); hold on;
    semilogy([info_riemna.time], optgap_riemna, '-*', 'color', colors{2}, 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    xlabel('Time (s)', 'fontsize', fs);
    ylabel('Optimality gap', 'fontsize', fs);
    legend({'RGD', 'RAGD', 'RNAG-C', 'RNAG-SC', 'RGD+RiemNA'}, 'fontsize', fs-5);
    
end
