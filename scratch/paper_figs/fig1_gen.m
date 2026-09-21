%% fig1.m
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));

%% Get IdVg data for Si nFET and OS nFET from the combined workbook
% (produced by scratch/compare_Si_vs_ITO.py's Excel export)

xlsxFile = fullfile(scriptDir, 'compare_Si_vs_ITO_data.xlsx');

siTable = readtable(xlsxFile, 'Sheet', 'Si nFET', 'VariableNamingRule', 'preserve');
osTable = readtable(xlsxFile, 'Sheet', 'OSFET', 'VariableNamingRule', 'preserve');

Vg_Si_nFET_full = siTable.("VGS (V)");
Id_Si_nFET_full = siTable.("ID (A)");

Vg_OS_nFET_full = osTable.("VGS (V)");
Id_OS_nFET_full = osTable.("ID (A)");

%% Figure 1: Plot IdVg for OS nFET for VTR illustration (SR/TR/AR schematic)

% Vg sweep is bidirectional (forward then reverse); keep only the forward
% sweep, then Vg >= 0 so the x-axis spans exactly [0, V_DD], matching the
% reference schematic's "0"/"V_DD" axis-extreme labels.
[~, pkIdx] = max(Vg_OS_nFET_full);
Vg_fwd = Vg_OS_nFET_full(1:pkIdx);
Id_fwd = Id_OS_nFET_full(1:pkIdx);

W = 100e-6; L = 2e-6; Vds = 0.05;

yLow = 1e-8;
yHigh = max(abs(Id_fwd) / W);

posMask = abs(Id_fwd) / W >= yLow & abs(Id_fwd) / W <= yHigh;
Vg_OS_nFET = Vg_fwd(posMask);
Id_OS_nFET = Id_fwd(posMask);

% V_T,off / V_T,on (linear-extrapolation extraction, from the workbook's
% Summary sheet) mark the TR's left/right boundaries.
summaryTable = readtable(xlsxFile, 'Sheet', 'Summary', 'VariableNamingRule', 'preserve');
osRow = strcmp(summaryTable.device, 'OSFET');
VTOFF = summaryTable.("VTOFF (V)")(osRow);
VTON = summaryTable.("VTON (V)")(osRow);

xmin = min(Vg_OS_nFET);
xmax = max(Vg_OS_nFET);

% Figure size matches the reference schematic's aspect ratio (~1.15:1,
% width:height).
fig1 = figure('Name','Id_vs_Vg_OS_nFET', 'Visible', 'off', 'Position', [100 100 500 420]);
ax = axes(fig1);
hold(ax, 'on');

% Set the label font size from the figure's OUTER pixel dimensions (fixed
% at figure-creation time) rather than the axes' inner plot-box dimensions
% (which aren't known yet -- they depend on the tight-layout inset below,
% which itself depends on the label font size). Deciding the font size
% first and letting TightInset react to it avoids that circularity; doing
% it the other way around (as a first attempt here did) computes the
% inset from the SMALL default font, then enlarges the font afterward, so
% labels end up overflowing the too-small margin that was reserved for
% them -- invisibly for a dummy/invisible title, "by luck" for a visible
% xlabel that just spills past its intended margin without being clipped.
ax.FontSize = min(fig1.Position(3), fig1.Position(4)) / 13.5;

% MATLAB's SVG export hardcodes font-family="Helvetica" regardless of
% what FontName is set to here or what actually rendered the PNG (verified
% empirically -- Helvetica/Arial/Trebuchet MS/Nimbus Sans all produce
% pixel-identical PNGs and the same SVG tag on this system), so this is
% purely documentation of intent for the SVG's declared font-family; it
% has no effect on the PNG or on layout/kerning.
ax.FontName = 'Helvetica';

% --- Background shading for SR / TR / AR --------------------------------
colorSR = [0.98 0.75 0.75];
colorTR = [0.80 0.72 0.93];
colorAR = [0.72 0.90 0.72];

yyaxis(ax, 'left'); % Left y-axis (log scale) -- shading is drawn here so
                     % its rectangle height is meaningful on a log axis
patch(ax, [xmin VTOFF VTOFF xmin], [yLow yLow yHigh yHigh], colorSR, ...
      'FaceAlpha', 0.6, 'EdgeColor', 'none');
patch(ax, [VTOFF VTON VTON VTOFF], [yLow yLow yHigh yHigh], colorTR, ...
      'FaceAlpha', 0.6, 'EdgeColor', 'none');
patch(ax, [VTON xmax xmax VTON], [yLow yLow yHigh yHigh], colorAR, ...
      'FaceAlpha', 0.6, 'EdgeColor', 'none');

% V_T,off / V_T,on dividers -- added after the patches but before the blue
% Id-Vg curves below, so they sit ABOVE the shading but BELOW both curves
% (MATLAB draws later-added objects on top of earlier ones). LineStyle is
% forced explicitly on every plot/semilogy call below: MATLAB's automatic
% per-series style cycling (SeriesIndex) silently varies LineStyle across
% successive calls on the same axes even when Color is given explicitly,
% which otherwise turns some of these solid lines into dashed/dotted ones.
plot(ax, [VTOFF VTOFF], [yLow yHigh], 'Color', [0.85 0 0], 'LineWidth', 3, 'LineStyle', '-');
plot(ax, [VTON VTON], [yLow yHigh], 'Color', [0 0.55 0], 'LineWidth', 3, 'LineStyle', '-');

semilogy(ax, Vg_OS_nFET, abs(Id_OS_nFET)/W, 'LineWidth', 3.5, 'Color', [0 0 1], 'LineStyle', '-');
ax.YAxis(1).Scale = 'log';  % semilogy doesn't reliably set the LEFT
                            % ruler's scale on a yyaxis-enabled axes --
                            % without this the "log" curve silently plots
                            % on a linear scale and exactly overlaps the
                            % right axis's linear curve.
% MATLAB's SVG export of 'tex'-interpreted subscripts (I_{D}) bakes in a
% fixed subscript offset instead of real glyph-width kerning, and its
% 'latex' interpreter (which lays subscripts out correctly) can't render
% sans-serif math (cmss is rejected), so neither built-in path gives
% correctly-kerned Helvetica-styled subscripts. Instead, xlabel/ylabel here
% are kept on the plain 'tex' interpreter but made invisible -- they exist
% only so their auto-computed TightInset reserves the right margin, exactly
% like the dummy title below -- and the VISIBLE label is built by hand out
% of separately-sized/positioned text() runs (see place_kerned_label at the
% end of this file), each queried for its real rendered Extent so the
% pieces butt up against each other with actual glyph-width kerning.
hYLabelLeft = ylabel(ax, 'I_{D} [log scale]', 'FontWeight', 'bold', 'Color', 'black');
ylim(ax, [yLow yHigh]);
yticks(ax, [])

yyaxis(ax, 'right'); % Right y-axis (linear scale)
plot(ax, Vg_OS_nFET, Id_OS_nFET/W, 'LineWidth', 3.5, 'Color', [0 0 1], 'LineStyle', '-');
hYLabelRight = ylabel(ax, 'I_{D} [linear scale]', 'FontWeight', 'bold', 'Color', 'black');
ylim(ax, [yLow yHigh]);
yticks(ax, [])

hXLabel = xlabel(ax, 'V_{GS}', 'FontWeight', 'bold', 'Color', 'black');

% Dummy (invisible) title: MATLAB collapses a title's reserved TightInset
% space to ~0 if its string is blank/whitespace, so a plain ' ' doesn't
% actually reserve anything. Reusing the xlabel's own string instead
% guarantees an identical text extent (same characters, same
% FontWeight/FontSize) -- i.e. the TOP margin this reserves is exactly as
% tall as the BOTTOM margin the real V_GS xlabel reserves.
title(ax, 'V_{GS}', 'FontWeight', 'bold', 'Color', 'none');

ax.FontWeight = "bold";
ax.LineWidth = 3.5;
ax.Box = 'on';  % draw all four sides (incl. top) at the same LineWidth,
                % not just the default left/bottom pair
ax.YAxis(1).Color = [0 0 0];
ax.YAxis(2).Color = [0 0 0];
xlim(ax, [xmin xmax])
xticks(ax, [])

% Use tight layout. ax.FontSize was already fixed (from the figure's outer
% pixel size) before the xlabel/ylabels/title above were created, so they
% all picked it up at creation time -- TightInset below reflects their
% real, final rendered size, not a to-be-replaced default.
set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02))

% Set the figure and axes background to transparent
set(gcf, 'Color', 'none');
set(ax, 'Color', 'none');

% Save the figure as a MATLAB .fig file in the script directory
savefig(fig1, fullfile(scriptDir, 'fig1.fig'));

exportgraphics(fig1, fullfile(scriptDir, 'fig1.pdf'), 'Padding', 'figure', 'ContentType', 'vector');
