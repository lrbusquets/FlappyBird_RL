%%
close all
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 12); % Set default font size to 12
set(groot, 'defaultLegendFontSize', 12); % Set default legend font size to 12
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
%set(groot, 'defaultAxesLabelInterpreter', 'latex');

%%
folders = {'relu', 'sigmoid'};
styles = {'-', '-.'};
colors = {"#0072BD", "#D95319"};

for i=1:length(folders)

    folder = folders{i};

    x = load(fullfile(folder,"benchmark_1_x.dat"));
    y = load(fullfile(folder,"benchmark_1_y.dat"));
    y_Adam = load(fullfile(folder,"benchmark_1_ypred_Adam.dat"));
    y_Basic = load(fullfile(folder,"benchmark_1_ypred_Basic.dat"));
    
    label_adam = strcat('Adam - ', folders{i});
    label_basic = strcat('Basic - ', folders{i});

    figure(1)
    plot(x, y, 'HandleVisibility', 'off', 'linew', 1, 'col', 'k')
    hold on
    plot(x, y_Adam, 'DisplayName', label_adam, 'linew', 1.5, 'col', colors{1}, 'linestyle', styles{i})
    plot(x, y_Basic, 'DisplayName', label_basic, 'linew', 1.5, 'col', colors{2}, 'linestyle', styles{i})
    xlabel ('$x ()$', 'interpreter', 'latex')
    ylabel ('$y ()$', 'interpreter', 'latex')
    legend('Location','best')
    
    MSE_adam = load(fullfile(folder,"benchmark_1_MSE_arrayAdam.dat"));
    std_adam = load(fullfile(folder, "benchmark_1_std_arrayAdam.dat"));
    std_basic = load(fullfile(folder, "benchmark_1_std_arraybasic.dat"));
    MSE_basic = load(fullfile(folder,"benchmark_1_MSE_arrayBasic.dat"));
    figure(2)
    semilogy(MSE_adam, 'DisplayName', label_adam, 'linew', 1.5, 'col', colors{1}, 'linestyle', styles{i})
    hold on
    semilogy(MSE_basic, 'DisplayName', label_basic, 'linew', 1.5, 'col', colors{2}, 'linestyle', styles{i})
    xlabel('Training epoch ()', 'interpreter', 'latex')
    ylabel('Training MSE ()', 'interpreter', 'latex')
    legend('Location','best')


    % Set the width and height of the current figure
    %set(gca, 'Position', [50, 50, 5000, 3500]);

end

for i=1:2
    figure(i)
    grid
    grid minor
end