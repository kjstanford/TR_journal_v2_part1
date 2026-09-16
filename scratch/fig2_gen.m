%% fig2.m
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));

%% Get IdVg data for Si nFET and OS nFET from the combined workbook
% (produced by scratch/compare_Si_vs_ITO.py's Excel export)

xlsxFile = fullfile(scriptDir, 'compare_Si_vs_ITO_data.xlsx');

siTable = readtable(xlsxFile, 'Sheet', 'Si nFET', 'VariableNamingRule', 'preserve');
osTable = readtable(xlsxFile, 'Sheet', 'OSFET', 'VariableNamingRule', 'preserve');

Vg_Si_nFET_full = siTable.("VGS (V)");
Id_Si_nFET_full = siTable.("ID (A)");
gm_Si_nFET_full = siTable.("gm (S)");
gm_by_Id_Si_nFET_full = siTable.("gm_over_id (1/V)");

Vg_OS_nFET_full = osTable.("VGS (V)");
Id_OS_nFET_full = osTable.("ID (A)");
gm_OS_nFET_full = osTable.("gm (S)");
gm_by_Id_OS_nFET_full = osTable.("gm_over_id (1/V)");

[~, pkIdx] = max(Vg_OS_nFET_full);
Vg_OS_nFET = Vg_OS_nFET_full(1:pkIdx);
Id_OS_nFET = Id_OS_nFET_full(1:pkIdx);
gm_OS_nFET = gm_OS_nFET_full(1:pkIdx);
gm_by_Id_OS_nFET = gm_by_Id_OS_nFET_full(1:pkIdx);
gm_OS_nFET_norm = gm_OS_nFET ./ max(gm_OS_nFET);

Wos = 100e-6; Los = 2e-6; Vds = 0.05;

[~, pkIdx] = max(Vg_Si_nFET_full);
Vg_Si_nFET = Vg_Si_nFET_full(1:pkIdx);
Id_Si_nFET = Id_Si_nFET_full(1:pkIdx);
gm_Si_nFET = gm_Si_nFET_full(1:pkIdx);
gm_by_Id_Si_nFET = gm_by_Id_Si_nFET_full(1:pkIdx);
gm_Si_nFET_norm = gm_Si_nFET ./ max(gm_Si_nFET);

Wsi = 10e-6; Lsi = 5e-6; Vds = 0.1;

%% Figure 2a: Plot gm/Id vs. Vg for OS nFET vs. Si nFET
fig2a = figure('Name','gm_over_id_vs_Vg_Si_vs_OS_nFET', 'Visible', 'off', 'Position', [100 100 420 450]);
ax = axes(fig2a);
hold(ax, 'on');

Id_OS_norm = abs(Id_OS_nFET) ./ max(gm_OS_nFET);
Id_Si_norm = abs(Id_Si_nFET) ./ max(gm_Si_nFET);

os_mask = gm_by_Id_OS_nFET <= log(10) / 0.06 & gm_by_Id_OS_nFET > log(10) / 0.2;
si_mask = gm_by_Id_Si_nFET <= log(10) / 0.06 & gm_by_Id_Si_nFET > log(10) / 0.2;

ax.FontSize = min(fig2a.Position(3), fig2a.Position(4)) / 20;
% 'Helvetica' isn't installed as an actual font file on this system --
% MATLAB's SVG export hardcodes whatever FontName is set here literally
% into the file, so a non-installed name forces every SVG viewer to
% substitute its own fallback (with its own metrics), causing text to
% render at a different size/proportion than the PNG, or not at all in
% stricter viewers. 'DejaVu Sans' is actually installed here, so the SVG
% and PNG stay visually consistent everywhere (see fig_1_2.m for the
% full writeup of this issue).
ax.FontName = 'Helvetica';

plot(ax, Id_OS_norm(os_mask), 1000 * log(10) ./ gm_by_Id_OS_nFET(os_mask), 'LineWidth', 3.5, 'Color', [0 0 1], 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0 0 1]);
plot(ax, Id_Si_norm(si_mask), 1000 * log(10) ./ gm_by_Id_Si_nFET(si_mask), 'LineWidth', 3.5, 'Color', [1 0 0], 'Marker', 's', 'MarkerSize', 8, 'MarkerFaceColor', [1 0 0]);

xlabel(ax, {'I_{D}', '[normalized]'}, 'FontWeight', 'bold');
ylabel(ax, {'SS', '[mV/dec]'}, 'FontWeight', 'bold');

% ax.YTickLabel = [];  % keep the y-axis numberless, but (unlike yticks(ax,[]))
%                       % leave the tick positions themselves in place so
%                       % major/minor tick marks below still have something
%                       % to draw
ylim(ax, [60 180]);
% yticks(ax, [50 100 150]);

% os_mask/si_mask select different numbers of points from the OS/Si
% sweeps, so Id_OS_norm/Id_Si_norm are column vectors of different
% lengths -- horzcat ([A, B]) requires matching row counts and errors,
% but vertcat ([A(:); B(:)]) only needs matching column counts (1 each
% here), so it works regardless of length.
xlim(ax, [min(Id_Si_norm(si_mask)), max(Id_OS_norm(os_mask))]);
xticks(ax, []);

ax.XScale = 'log';
ax.FontWeight = "bold";
ax.LineWidth = 3.5;
ax.Box = 'on';
% ax.XAxis.MinorTick = 'on';
ax.YAxis.MinorTick = 'on';
ax.TickLength = [0.025 0.025];  % taller major ticks (default ~[0.01 0.025]);
                                 % minor ticks scale proportionally shorter

set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02))
set(gcf, 'Color', 'none');
set(ax, 'Color', 'none');

savefig(fig2a, fullfile(scriptDir, 'fig2a.fig'));

%% Figure 2b: Plot muFE (\propto gm) vs. Vg for OS nFET vs. Si nFET
fig2 = figure('Name','muFE_vs_Vg_Si_vs_OS_nFET', 'Visible', 'off', 'Position', [100 100 420 450]);
ax = axes(fig2);
hold(ax, 'on');

ax.FontSize = min(fig2.Position(3), fig2.Position(4)) / 20;
% 'Helvetica' isn't installed as an actual font file on this system --
% MATLAB's SVG export hardcodes whatever FontName is set here literally
% into the file, so a non-installed name forces every SVG viewer to
% substitute its own fallback (with its own metrics), causing text to
% render at a different size/proportion than the PNG, or not at all in
% stricter viewers. 'DejaVu Sans' is actually installed here, so the SVG
% and PNG stay visually consistent everywhere (see fig_1_2.m for the
% full writeup of this issue).
ax.FontName = 'Helvetica';
plot(ax, Vg_OS_nFET, gm_OS_nFET_norm, 'LineWidth', 3.5, 'Color', [0 0 1]);
plot(ax, Vg_Si_nFET, gm_Si_nFET_norm, 'LineWidth', 3.5, 'Color', [1 0 0]);

xlabel(ax, 'V_{GS} [V]', 'FontWeight', 'bold');
ylabel(ax, {'\mu_{FE}', '[normalized]'}, 'FontWeight', 'bold');

% ax.YTickLabel = [];  % keep the y-axis numberless, but (unlike yticks(ax,[]))
%                       % leave the tick positions themselves in place so
%                       % major/minor tick marks below still have something
%                       % to draw
ylim(ax, [0 1.05]);
yticks(ax, []);

ax.FontWeight = "bold";
ax.LineWidth = 3.5;
ax.Box = 'on';
ax.XAxis.MinorTick = 'on';
% ax.YAxis.MinorTick = 'on';
ax.TickLength = [0.025 0.025];  % taller major ticks (default ~[0.01 0.025]);
                                 % minor ticks scale proportionally shorter

set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02))
set(gcf, 'Color', 'none');
set(ax, 'Color', 'none');

savefig(fig2, fullfile(scriptDir, 'fig2b.fig'));