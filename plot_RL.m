%%
close all
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 12); % Set default font size to 12
set(groot, 'defaultLegendFontSize', 12); % Set default legend font size to 12
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
%set(groot, 'defaultAxesLabelInterpreter', 'latex');

score = load('score_curve_Helena');
reward = load('reward_curve_Helena');

figure
plot(score,'LineWidth',1.5)
ylabel('Score ()','interpreter','latex')
xlabel('\# Game','interpreter','latex')
grid
grid minor

figure
plot(reward,'LineWidth',1.5,'col', "#D95319")
ylabel('Reward ()','interpreter','latex')
xlabel('\# Game','interpreter','latex')
grid
grid minor

