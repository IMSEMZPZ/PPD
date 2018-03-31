%   Algorithm:
%   1: cocoa
%   2: ppd


%% Set Params
file_name = 'kddb.txt';

C = 0.00001;

[a,b,c,d,e,f] = draw_av(file_name, C);
save './cocoa_different_block' a b c d e f
[a,b,c,d,e,f] = draw_ad(file_name, C);
save './cocoa_plus_different_block' a b c d e f
[a,b,c,d,e,f] = draw_psdca(file_name, C);
save './psdca_different_block' a b c d e f



function [Cocoa_average_dual_gap_32, Cocoa_average_dual_gap_64, Cocoa_average_dual_gap_128, Cocoa_average_dual_gap_256, Cocoa_average_dual_gap_512, Cocoa_average_dual_gap_1024] = draw_av(file_name, C)
    %%
    loss = 'L2_svm';
    tol = 1e-12;
    n_epoch = 40;
    verbose = 1;
    n_thread = 32;
    batch_size = 8;
    H = 10;
    gamma = 0.333;
    use_best_gamma = 0;
    algorithm = 'cocoa';
    %% Cocoa-averaging
    n_block = 32;
    [Cocoa_average_primal_val_32, Cocoa_average_dual_val_32, Cocoa_average_dual_gap_32] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-averaging
    n_block = 64;
    [Cocoa_average_primal_val_64, Cocoa_average_dual_val_64, Cocoa_average_dual_gap_64] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-averaging
    n_block = 128;
    [Cocoa_average_primal_val_128, Cocoa_average_dual_val_128, Cocoa_average_dual_gap_128] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
     %% Cocoa-averaging
    n_block = 256;
    [Cocoa_average_primal_val_256, Cocoa_average_dual_val_256, Cocoa_average_dual_gap_256] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-averaging
    n_block = 512;
    [Cocoa_average_primal_val_512, Cocoa_average_dual_val_512, Cocoa_average_dual_gap_512] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    n_block = 1024;
    [Cocoa_average_primal_val_1024, Cocoa_average_dual_val_1024, Cocoa_average_dual_gap_1024] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    %% Draw Dual_gap
    figure,semilogy(0:size(Cocoa_average_dual_gap_32)-1, abs(Cocoa_average_dual_gap_32), 'r-*',...
        0:size(Cocoa_average_dual_gap_64)-1, abs(Cocoa_average_dual_gap_64), 'g-x',...
        0:size(Cocoa_average_dual_gap_128)-1, abs(Cocoa_average_dual_gap_128), 'b-s',...
        0:size(Cocoa_average_dual_gap_256)-1, abs(Cocoa_average_dual_gap_256), 'c-o', ...
        0:size(Cocoa_average_dual_gap_512)-1, abs(Cocoa_average_dual_gap_512), 'm-+', ...
        0:size(Cocoa_average_dual_gap_1024)-1, abs(Cocoa_average_dual_gap_1024), 'k-v');
    legend('k = 32', 'k = 64', 'k = 128', 'k = 256', 'k = 512','k = 1024');
    xlabel({'Epoch'});
    ylabel({'Dual-Gap'})
    if(strcmp(file_name,'rcv1_test.binary'))
        file_name = 'rcv1\_test.binary';
    end
    if(strcmp(file_name, 'url_combined'))
            file_name = 'url\_combined';
    end
    if(strcmp(file_name, 'heart_scale'))
            file_name = 'heart\_scale';
    end   
    label = [file_name, ', CoCoA'];
    title(label);
end
function [Cocoa_add_dual_gap_32, Cocoa_add_dual_gap_64, Cocoa_add_dual_gap_128, Cocoa_add_dual_gap_256,  Cocoa_add_dual_gap_512, Cocoa_add_dual_gap_1024] = draw_ad(file_name, C)
    %%
    loss = 'L2_svm';
    tol = 1e-12;
    n_epoch = 40;
    verbose = 1;
    n_thread = 32;
    batch_size = 8;
    H = 10;
    gamma = 0.333;
    use_best_gamma = 0;
    algorithm = 'cocoa';
    %% Cocoa-adding
    n_block = 32;
    [Cocoa_add_primal_val_32, Cocoa_add_dual_val_32, Cocoa_add_dual_gap_32] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 64;
    [Cocoa_add_primal_val_64, Cocoa_add_dual_val_64, Cocoa_add_dual_gap_64] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 128;
    [Cocoa_add_primal_val_128, Cocoa_add_dual_val_128, Cocoa_add_dual_gap_128] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
     %% Cocoa-adding
    n_block = 256;
    [Cocoa_add_primal_val_256, Cocoa_add_dual_val_256, Cocoa_add_dual_gap_256] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 512;
    [Cocoa_add_primal_val_512, Cocoa_add_dual_val_512, Cocoa_add_dual_gap_512] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    n_block = 1024;
    [Cocoa_add_primal_val_1024, Cocoa_add_dual_val_1024, Cocoa_add_dual_gap_1024] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    %% Draw Dual_gap
    figure,semilogy(0:size(Cocoa_add_dual_gap_32)-1, abs(Cocoa_add_dual_gap_32), 'r-*',...
        0:size(Cocoa_add_dual_gap_64)-1, abs(Cocoa_add_dual_gap_64), 'g-x',...
        0:size(Cocoa_add_dual_gap_128)-1, abs(Cocoa_add_dual_gap_128), 'b-s',...
        0:size(Cocoa_add_dual_gap_256)-1, abs(Cocoa_add_dual_gap_256), 'c-o', ...
        0:size(Cocoa_add_dual_gap_512)-1, abs(Cocoa_add_dual_gap_512), 'm-+', ...
        0:size(Cocoa_add_dual_gap_1024)-1, abs(Cocoa_add_dual_gap_1024), 'k-v');
    legend('k = 32', 'k = 64', 'k = 128', 'k = 256', 'k = 512','k = 1024');
    xlabel({'Epoch'});
    ylabel({'Dual-Gap'})
    if(strcmp(file_name,'rcv1_test.binary'))
        file_name = 'rcv1\_test.binary';
    end
    if(strcmp(file_name, 'url_combined'))
            file_name = 'url\_combined';
    end
    if(strcmp(file_name, 'heart_scale'))
            file_name = 'heart\_scale';
    end   
    label = [file_name, ', CoCoA+'];
    title(label);
end


function [PPD_dual_gap_32, PPD_dual_gap_64, PPD_dual_gap_128, PPD_dual_gap_256,  PPD_dual_gap_512, PPD_dual_gap_1024] = draw_psdca(file_name, C)
    %%
    loss = 'L2_svm';
    tol = 1e-12;
    n_epoch = 40;
    verbose = 1;
    n_thread = 32;
    batch_size = 8;
    H = 10;
    gamma = 0.333;
    use_best_gamma = 0;
    algorithm = 'ppd';
    %% Cocoa-adding
    n_block = 32;
    [PPD_primal_val_32, PPD_dual_val_32, PPD_dual_gap_32] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 64;
    [PPD_primal_val_64, PPD_dual_val_64, PPD_dual_gap_64] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 128;
    [PPD_primal_val_128, PPD_dual_val_128, PPD_dual_gap_128] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
     %% Cocoa-adding
    n_block = 256;
    [PPD_primal_val_256, PPD_dual_val_256, PPD_dual_gap_256] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    %% Cocoa-adding
    n_block = 512;
    [PPD_primal_val_512, PPD_dual_val_512, PPD_dual_gap_512] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    
    n_block = 1024;
    [PPD_primal_val_1024, PPD_dual_val_1024, PPD_dual_gap_1024] = ...
        Interface(file_name, loss, algorithm, C, tol, n_epoch, verbose, n_thread, batch_size, n_block, H, gamma, use_best_gamma);
    %% Draw Dual_gap
    figure,semilogy(0:size(PPD_dual_gap_32)-1, abs(PPD_dual_gap_32), 'r-*',...
        0:size(PPD_dual_gap_64)-1, abs(PPD_dual_gap_64), 'g-x',...
        0:size(PPD_dual_gap_128)-1, abs(PPD_dual_gap_128), 'b-s',...
        0:size(PPD_dual_gap_256)-1, abs(PPD_dual_gap_256), 'c-o', ...
        0:size(PPD_dual_gap_512)-1, abs(PPD_dual_gap_512), 'm-+', ...
        0:size(PPD_dual_gap_1024)-1, abs(PPD_dual_gap_1024), 'k-v');
    legend('k = 32', 'k = 64', 'k = 128', 'k = 256', 'k = 512','k = 1024');
    xlabel({'Epoch'});
    ylabel({'Dual-Gap'})
    if(strcmp(file_name,'rcv1_test.binary'))
        file_name = 'rcv1\_test.binary';
    end
    if(strcmp(file_name, 'url_combined'))
            file_name = 'url\_combined';
    end
    if(strcmp(file_name, 'heart_scale'))
            file_name = 'heart\_scale';
    end   
    label = [file_name, ', PPD'];
    title(label);
end


