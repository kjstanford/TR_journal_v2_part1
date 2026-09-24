%% fig2.m
clear; clc; close all;

% scriptDir = fileparts(mfilename('fullpath'));
scriptDir = '/home/kj2011/Papers/TR_journal_v2_part1/scratch/paper_figs';

%% Get IdVg data for Si nFET and OS nFET from the combined workbook
% (produced by scratch/compare_Si_vs_ITO.py's Excel export)

xlsxFile = fullfile(scriptDir, 'compare_Si_vs_ITO_data.xlsx');

osTable = readtable(xlsxFile, 'Sheet', 'OSFET', 'VariableNamingRule', 'preserve');
summaryTable = readtable(xlsxFile, 'Sheet', 'Summary', 'VariableNamingRule', 'preserve');

Vg_OS_nFET_full = osTable.("VGS (V)");
Id_OS_nFET_full = osTable.("ID (A)");
gm_OS_nFET_full = osTable.("gm (S)");
gm_by_Id_OS_nFET_full = osTable.("gm_over_id (1/V)");

[~, pkIdx] = max(Vg_OS_nFET_full);
Vg_OS_nFET = Vg_OS_nFET_full(1:pkIdx);
Id_OS_nFET = Id_OS_nFET_full(1:pkIdx);
gm_OS_nFET = gm_OS_nFET_full(1:pkIdx);
gm_by_Id_OS_nFET = gm_by_Id_OS_nFET_full(1:pkIdx);

Wos = 100e-6; Los = 2e-6; Vds = 0.05;

%% VTR extraction summary numbers for the OS FET (pre-computed by
% scratch/compare_Si_vs_ITO.py and exported into the "Summary" sheet --
% this script only re-derives the two extrapolation lines used to
% illustrate how V_T,off/V_T,on were obtained, not the extraction itself)
osRow = summaryTable(strcmp(summaryTable.device, 'OSFET'), :);
VTON = osRow.("VTON (V)");
VTOFF = osRow.("VTOFF (V)");
VTR = osRow.("VTR (V)");
ID0 = osRow.("ID0 (A)");
idx_on = osRow.("idx_on")+1;
idx_off = osRow.("idx_off")+1;
off_slope = osRow.("off_slope");
off_intercept = osRow.("off_intercept");
on_slope = osRow.("on_slope");
on_intercept = osRow.("on_intercept");

%% Figure 5a: illustrate the log/linear extrapolation VTR extraction
% method on the OS nFET Id-Vg curve
fig4e = figure('Name','VTR_extraction_illustration_OS_nFET', 'Visible', 'off', 'Position', [100 100 650 400]);
ax = axes(fig4e);
hold(ax, 'on');

ax.FontSize = min(fig4e.Position(3), fig4e.Position(4)) / 15;
% ax.FontSize = 20;
ax.FontName = 'Arial';



log_fit_line_X = linspace(Vg_OS_nFET(idx_off), Vg_OS_nFET(end), 1000);
log_fit_line_X = log_fit_line_X(log_fit_line_X <= VTOFF);
log_fit_line_Y = off_slope * log_fit_line_X + off_intercept;

xline(ax, VTON, 'LineWidth', 3, 'Color', [0 1 0], 'LineStyle', '-');
xline(ax, VTOFF, 'LineWidth', 3, 'Color', [1 0 0], 'LineStyle', '-');
yyaxis(ax, 'left');
plot(ax, Vg_OS_nFET, log10(abs(Id_OS_nFET)), 'LineWidth', 3.5, 'Color', [0 0 1], 'LineStyle', '-');
yline(ax, log10(ID0), 'LineWidth', 3, 'Color', '#A52A2A', 'LineStyle', '--');
plot(ax, log_fit_line_X, log_fit_line_Y, 'LineWidth', 3.5, 'Color', [1 0 0], 'LineStyle', '--');
plot(ax, Vg_OS_nFET(idx_off), log10(abs(Id_OS_nFET(idx_off))), 'o', 'MarkerSize', 10, 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', [1 0 0]);
plot(ax, log_fit_line_X(end), log_fit_line_Y(end), 'o', 'MarkerSize', 10, 'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', [1 0 0]);
hYLabelLeft = ylabel(ax, 'I_{D} [log scale]', 'FontWeight', 'bold', 'Color', 'black');
yticks(ax, [])
ylim(ax, [-14 log10(max(abs(Id_OS_nFET)))]);

yyaxis(ax, 'right');
lin_fit_line_X = linspace(Vg_OS_nFET(1), Vg_OS_nFET(idx_on), 1000);
lin_fit_line_Y = on_slope * lin_fit_line_X + on_intercept;
lin_fit_line_X = lin_fit_line_X(lin_fit_line_Y >= 0);
lin_fit_line_Y = lin_fit_line_Y(lin_fit_line_Y >= 0);
plot(ax, Vg_OS_nFET, Id_OS_nFET, 'LineWidth', 3.5, 'Color', [0 0 1], 'LineStyle', '-');
yline(ax, 0, 'LineWidth', 3, 'Color', '#A52A2A', 'LineStyle', '--');
plot(ax, lin_fit_line_X, lin_fit_line_Y, 'LineWidth', 3.5, 'Color', [0 1 0], 'LineStyle', '--');
plot(ax, Vg_OS_nFET(idx_on), abs(Id_OS_nFET(idx_on)), 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0 1 0], 'MarkerEdgeColor', [0 1 0]);
plot(ax, lin_fit_line_X(1), lin_fit_line_Y(1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0 1 0], 'MarkerEdgeColor', [0 1 0]);
hYLabelRight = ylabel(ax, 'I_{D} [linear scale]', 'FontWeight', 'bold', 'Color', 'black');
yticks(ax, [])
ylim(ax, [0 max(Id_OS_nFET)]);

hXLabel = xlabel(ax, 'V_{GS}', 'FontWeight', 'bold', 'Color', 'black');
xticks(ax, [])

ax.FontWeight = "bold";
ax.LineWidth = 3.5;
ax.Box = 'on';
ax.YAxis(1).Color = [0 0 0];
ax.YAxis(2).Color = [0 0 0];

savefig(fig4e, fullfile(scriptDir, 'fig4e.fig'));
exportgraphics(fig4e, fullfile(scriptDir, 'fig4e.pdf'), 'Padding', 'figure', 'ContentType', 'vector');
