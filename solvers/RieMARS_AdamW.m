function [x, info, options] = RieMARS_AdamW(problem, x, options)
% Riemannian MARS-AdamW optimizer with stochastic gradients for Manopt.
%
% function [x, info, options] = RieMARS_AdamW(problem)
% function [x, info, options] = RieMARS_AdamW(problem, x0)
% function [x, info, options] = RieMARS_AdamW(problem, x0, options)
% function [x, info, options] = RieMARS_AdamW(problem, [], options)
%
% Apply the Riemannian MARS-AdamW optimizer to the problem defined
% in the problem structure, starting at x0 if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x0 as [] (the empty matrix).
%
% The problem structure must contain the following fields:
%
%   problem.M
%       Defines the manifold to optimize over, given by a factory.
%
%   problem.ncostterms
%       An integer specifying how many terms are in the cost function.
%
%   problem.partialgrad or problem.partialegrad
%       Describes the partial gradients of the cost function. If the cost
%       function is of the form f(x) = sum_{k=1}^N f_k(x),
%       then partialegrad(x, K) = sum_{k \in K} grad f_k(x).
%
% Some of the options of the solver are specific to this file.
% Please have a look inside the code.
%
% See also: RSGD

% This file is part of Manopt: www.manopt.org.
% Original author: [Your Name], [Date].

    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', 'No partial gradient provided. The algorithm will likely abort.');
    end

    % Set local defaults
    localdefaults.maxiter = 1000;
    localdefaults.lr = 0.001;
    localdefaults.scheduler = 'linear';
    localdefaults.beta1 = 0.9;
    localdefaults.beta2 = 0.999;
    localdefaults.epsilon = 1e-8;
    localdefaults.weight_decay = 0.0;
    localdefaults.gamma = 0.025;
    localdefaults.batchsize = 32;    % Mini-batch size
    localdefaults.verbosity = 2;
    localdefaults.tolgradnorm = 1e-20;
    localdefaults.checkperiod = 100; % Frequency of stopping criteria checks

    % Merge global and local defaults, then merge with user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);

    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end

    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();

    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    grad_cnt = 0;

    % Initialize state variables
    exp_avg = problem.M.zerovec(x); % First moment estimate (tangent vector)
    exp_avg_sq = 0.0; % Second moment estimate (scalar)
    stepsize = NaN;

    % Initialize last_grad and x_prev
    last_grad = problem.M.zerovec(x);
    x_prev = x;

    % Initialize iteration counter
    iter = 0;
    savedstats = 0;
    elapsed_time = 0;

    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    if isinf(options.maxiter)
        % We trust that if the user set maxiter = inf, then they defined
        % another stopping criterion.
        preallocate = 1e5;
    else
        preallocate = ceil(options.maxiter / options.checkperiod) + 1;
    end
    info(preallocate).iter = [];

    % Display information header for the user.
    if options.verbosity >= 1
        fprintf('\n-------------------------------------------------------\n');
        fprintf('Riemannian MARS-AdamW:  iter\t               cost val\t    grad. norm\t stepsize\n');
        fprintf('Riemannian MARS-AdamW:  %5d\t%+.16e\t%.8e\n', 0, cost, gradnorm);
    end

    % Define the scheduler
    if strcmp(options.scheduler, 'fixed')
        lr = @(initial_lr, current_iter, max_iter) cosine_lr(initial_lr, current_iter, max_iter);
    elseif strcmp(options.scheduler, 'cosine')
        lr = @(initial_lr, current_iter, max_iter) cosine_lr(initial_lr, current_iter, max_iter);
    elseif strcmp(options.scheduler, 'linear')
        lr = @(initial_lr, current_iter, max_iter) linear_lr(initial_lr, current_iter, max_iter);
    elseif strcmp(options.scheduler, 'exponential')
        lr = @(initial_lr, current_iter, max_iter) exponential_lr(initial_lr, current_iter, max_iter);
    else
        error('Unknown scheduler type');
    end

    % Main loop
    while iter < options.maxiter

        iter = iter + 1;

        if strcmp(options.scheduler, 'fixed')
            learning_rate = options.lr;
        else
            learning_rate = lr(options.lr, iter, options.maxiter);
        end

        % Record start time.
        start_time = tic();

        % Draw random indices for the mini-batch
        idx_batch = randi(problem.ncostterms, options.batchsize, 1);

        % Compute partial gradient at current point
        grad = getPartialGradient(problem, x, idx_batch, storedb, key);
        gradnorm = problem.M.norm(x, grad);

        grad_cnt = grad_cnt + options.batchsize;

        % Compute c_t
        if iter > 1
            % Transport last_grad from x_prev to x
            transported_last_grad = problem.M.transp(x_prev, x, last_grad);
            % Compute grad_diff = grad - transported_last_grad
            grad_diff = problem.M.lincomb(x, 1, grad, -1, transported_last_grad);
            % grad_diff = grad - transported_last_grad;
        else
            % For the first iteration, grad_diff is zero
            grad_diff = problem.M.zerovec(x);
        end

        % Compute c_t = grad + coeff * grad_diff
        coeff = options.gamma * (options.beta1 / (1 - options.beta1));
        c_t = grad + (grad_diff * coeff);
        % c_t = problem.M.lincomb(x, 1, grad, coeff, grad_diff);
        c_t_norm_sq = problem.M.norm(x, c_t)^2;
        % c_t = grad;

        % c_t_norm = problem.M.norm(x, c_t);
        % if c_t_norm > 1
        %     c_t = problem.M.lincomb(x, 1 / c_t_norm, c_t);
        % end

        % Update biased first moment estimate (exp_avg)
        exp_avg = problem.M.lincomb(x, options.beta1, exp_avg, (1 - options.beta1), c_t);

        % Compute the squared norm of c_t
        % c_t_norm_sq = problem.M.norm(x, c_t);
        % assert(c_t_norm_sq >= 0, 'Squared norm of c_t is negative: %.6e', c_t_norm_sq);

        % Update biased second moment estimate (scalar)
        % exp_avg_sq = problem.M.lincomb(x, options.beta2, exp_avg_sq, (1 - options.beta2), c_t_norm_sq);
        exp_avg_sq = options.beta2 * exp_avg_sq + (1 - options.beta2) * c_t_norm_sq;

        % Compute bias-corrected first and second moment estimates
        bias_correction1 = 1 - options.beta1^iter;
        bias_correction2 = 1 - options.beta2^iter;

        exp_avg_hat = problem.M.lincomb(x, 1 / bias_correction1, exp_avg);
        % exp_avg_hat = exp_avg;
        exp_avg_sq_hat = exp_avg_sq / bias_correction2;
        % exp_avg_sq_hat = exp_avg_sq;

        % Compute the deinator (scalar)
        denom = sqrt(exp_avg_sq_hat) + options.epsilon;

        if mod(iter, options.checkperiod) == 0
            % fprintf('Iter: %5d, grad = %.4e, denom = %.4e, exp_avg_hat = %.4e, exp_avg_sq_hat = %.4e\n', iter, grad, denom, exp_avg_hat, exp_avg_sq_hat);
        end

        % Compute update direction
        scaled_grad = exp_avg_hat /denom;
        % scaled_grad = problem.M.lincomb(x, 1, exp_avg_hat, 0, 0) ./ denom;

        % Apply weight decay (if any)
        if options.weight_decay ~= 0
            % Compute Euclidean gradient of weight decay term
            egrad_weight_decay = options.weight_decay * x;
            % Convert to Riemannian gradient
            weight_decay_grad = problem.M.egrad2rgrad(x, egrad_weight_decay);
            % Compute update direction
            weight_decay_update = problem.M.lincomb(x, options.lr, weight_decay_grad);
            % Combine with scaled_grad
            update_direction = problem.M.lincomb(x, -learning_rate, scaled_grad, -1, weight_decay_update);
        else
            update_direction = problem.M.lincomb(x, -learning_rate, scaled_grad);
        end

        % Retraction: x_new = M.retr(x, update_direction)
        x_new = problem.M.retr(x, update_direction);

        % Transport exp_avg to the new tangent space
        exp_avg = problem.M.transp(x, x_new, exp_avg);

        % Update last_grad and x_prev
        last_grad = grad;
        x_prev = x;
        stepsize = problem.M.dist(x_prev, x_new);

        % Update x
        x = x_new;
        key = storedb.getNewKey(); % Update key for the new x

        % Elapsed time for this iteration
        elapsed_time = elapsed_time + toc(start_time);

        % Check stopping criteria and save stats every checkperiod iterations.
        if mod(iter, options.checkperiod) == 0

            % Compute full cost and gradient norm for logging
            [cost, fullgrad] = getCostGrad(problem, x, storedb, key);
            fullgradnorm = problem.M.norm(x, fullgrad);

            % Log statistics
            stats = savestats();
            info(savedstats+1) = stats;
            savedstats = savedstats + 1;

            % Reset timer.
            elapsed_time = 0;

            % Display information
            if options.verbosity >= 1
                fprintf('Riemannian MARS-AdamW:  %5d\t%+.16e\t%.8e\t%.8e\n', iter, cost, fullgradnorm, stepsize);
            elseif options.verbosity >= 2
                    fprintf('Riemannian MARS-AdamW: Iter %d:\n', iter);
                    % fprintf('  c_t_norm_sq = %.6e\n', c_t_norm_sq);
                    fprintf('  exp_avg_sq = %.6e\n', exp_avg_sq);
                    fprintf('  exp_avg_sq_hat = %.6e\n', exp_avg_sq_hat);
                    fprintf('  denom = %.6e\n', denom);
                    fprintf('  ||scaled_grad|| = %.6e\n', problem.M.norm(x, scaled_grad));
                    fprintf('  stepsize = %.6e\n', stepsize);
                    fprintf('  learning rate = %.6e\n', learning_rate)
                    fprintf('  cost = %.6e\n', cost)
            end

            % Run standard stopping criterion checks.
            [stop, reason] = stoppingcriterion(problem, x, options, info, savedstats);
            if stop
                if options.verbosity >= 1
                    fprintf([reason '\n']);
                end
                break;
            end

        end

    end

    % Keep only the relevant portion of the info struct-array.
    info = info(1:savedstats);

    % Helper function to collect statistics to be saved at index savedstats+1 in info.
    function stats = savestats()
        stats.iter = iter;
        if savedstats == 0
            stats.time = 0;
            stats.stepsize = stepsize;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end

    function lr = cosine_lr(initial_lr, current_iter, max_iter)
        lr = initial_lr * 1 * (1 + cos(pi * current_iter / max_iter))^2 * exp(-log(2)/max_iter * current_iter);
    end

    function lr = linear_lr(initial_lr, current_iter, max_iter)
        final_lr = 1e-20; % Target final learning rate
        lr = initial_lr - current_iter * (initial_lr - final_lr) / max_iter;
    end

    function lr = exponential_lr(initial_lr, current_iter, max_iter)
        final_lr = 1e-20; % Target final learning rate
        decay_rate = log(final_lr / initial_lr) / max_iter; % Compute decay rate
        lr = initial_lr * exp(decay_rate * current_iter);
    end

end
