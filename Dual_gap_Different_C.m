%   Algorithm:
%	1: Serial
%   2: Mini_batch
%   3: Passcode
%   4: Cocoa
%   5: Parallel_SDCA


%% Set Params
file_name = 'rcv1_test.binary';
[a,b,c] = draw(file_name, 0.0001);
save './rcv1_test_C_0001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.00001);
save './rcv1_test_C_00001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.000001);
save './rcv1_test_C_000001_dif_alg' a b c

file_name = 'kddb.t';
[a,b,c] = draw(file_name, 0.0001);
save './kdd_t_C_0001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.00001);
save './kdd_t_C_00001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.000001);
save './kdd_t_C_000001_dif_alg' a b c

file_name = 'news20.binary';
[a,b,c] = draw(file_name, 0.0001);
save './news20_C_0001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.00001);
save './news20_C_00001_dif_alg' a b c
[a,b,c] = draw(file_name, 0.000001);
save './news20_C_000001_dif_alg' a b c

function [Cocoa_average_dual_gap, Cocoa_add_dual_gap, Parallel_SDCA_dual_gap] = draw(file_name, C)
    %%
    loss = 'L2_svm';
    tol = 1e-12;
    n_epoch = 40;
    verbose = 1;
    n_thread = 32;
    batch_size = 8;
    n_block = 64;
    H = 10;
    gamma = 0.333;
    use_best_gamma = 0;
    %% Cocoa-averaging
    algorithm = 'cocoa';
    [Cocoa_average_primal_val, Cocoa_average_dual_val, Cocoa_average_dual_gap] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);

    %% Cocoa-adding
    gamma = 1;
    algorithm = 'cocoa';
    [Cocoa_add_primal_val, Cocoa_add_dual_val, Cocoa_add_dual_gap] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);

    %% Parallel_SDCA
    gamma = 1;
    use_best_gamma = 1;
    algorithm = 'parallel_sdca';
    [Parallel_SDCA_primal_val, Parallel_SDCA_dual_val, Parallel_SDCA_dual_gap] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);

    %% Draw Dual_gap
    figure,semilogy(0:size(Cocoa_average_dual_gap)-1, abs(Cocoa_average_dual_gap), 'r-*',...
        0:size(Cocoa_add_dual_gap)-1, abs(Cocoa_add_dual_gap), 'b-o',...
        0:size(Parallel_SDCA_dual_gap)-1, abs(Parallel_SDCA_dual_gap), 'k-v');
    legend('Cocoa','Cocoa +','PPD');
    xlabel({'Epoch'});
    ylabel({'Dual-Gap'})
    label = ', C = ';
    label2 = num2str(C);
    if(strcmp(file_name,'rcv1_test.binary'))
        file_name = 'rcv1\_test.binary';
    end
    if(strcmp(file_name, 'url_combined'))
            file_name = 'url\_combined';
    end
    if(strcmp(file_name, 'heart_scale'))
            file_name = 'heart\_scale';
    end
        
    label = [file_name,label, label2];
    title(label);
end



