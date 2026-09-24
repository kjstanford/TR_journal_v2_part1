%% plot_IdVg_all.m
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));

%% Load all IdVg data
% Columns are named {DieR}_{DieC}_{DeviceID}_Vg and
% {DieR}_{DieC}_{DeviceID}_Id_{Vd}_{cyclenum}

csvFile = fullfile(scriptDir, 'ITO_t2nm_L2um_W100um_compiled.csv');
T = readtable(csvFile, 'VariableNamingRule', 'preserve');
varNames = T.Properties.VariableNames;

vgIdx = find(endsWith(varNames, '_Vg'));

%% Figure: IdVg for all devices/cycles, Vds = 0.05 V (red) vs 1.5 V (blue)
fig = figure('Name', 'IdVg_all', 'Visible', 'off', 'Position', [100 100 650 500]);
ax = axes(fig);
hold(ax, 'on');

% ax.FontSize = min(fig.Position(3), fig.Position(4)) / 22;
ax.FontSize = 20;
ax.FontName = 'Helvetica';

h_low = gobjects(0);
h_high = gobjects(0);

for k = 1:numel(vgIdx)
    vgName = varNames{vgIdx(k)};
    devPrefix = extractBefore(vgName, '_Vg');
    Vg_full = T.(vgName);

    % Vg is a double (forward + backward) sweep; keep the forward leg only
    [~, pkIdx] = max(Vg_full);
    Vg = Vg_full(1:pkIdx);

    idLowNames = varNames(startsWith(varNames, [devPrefix '_Id_0.05_']));
    idHighNames = varNames(startsWith(varNames, [devPrefix '_Id_1.5_']));

    for j = 1:numel(idLowNames)
        Id = T.(idLowNames{j});
        h = plot(ax, Vg, Id(1:pkIdx), 'LineWidth', 2.5, 'Color', [1 0 0]);
        h.Color(4) = 0.5;
        h_low(end+1) = h; 
    end

    for j = 1:numel(idHighNames)
        Id = T.(idHighNames{j});
        h = plot(ax, Vg, Id(1:pkIdx), 'LineWidth', 2.5, 'Color', [0 0 1]);
        h.Color(4) = 0.5;
        h_high(end+1) = h; 
    end
end

xlabel(ax, {'V_{GS} [V]'}, 'FontWeight', 'bold');
ylabel(ax, {'I_D [A]'}, 'FontWeight', 'bold');
xlim(ax, [0 3]);
ylim(ax, [1e-11 2e-3]);
yticks(ax, [1e-11 1e-9 1e-7 1e-5 1e-3]);

ax.YScale = 'log';
ax.FontWeight = 'bold';
ax.LineWidth = 3.5;
ax.Box = 'on';
ax.XAxis.MinorTick = 'on';
ax.YAxis.MinorTick = 'on';
% ax.TickLength = [0.025 0.025];  % taller major ticks (default ~[0.01 0.025]);
                                 % minor ticks scale proportionally shorter

lgd = legend([h_low(1), h_high(1)], {'V_{DS} = 0.05 V', 'V_{DS} = 1.5 V'}, 'Location', 'best');
% lgd.Box = 'off';

% set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02))
set(gcf, 'Color', 'none');
set(ax, 'Color', 'none');

savefig(fig, fullfile(scriptDir, 'IdVg_all.fig'));
exportgraphics(fig, fullfile(scriptDir, 'IdVg_all.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(scriptDir, 'IdVg_all.png'), 'Resolution', 300);
