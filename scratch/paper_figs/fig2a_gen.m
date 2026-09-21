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

ax.FontSize = min(fig2a.Position(3), fig2a.Position(4)) / 18;
% MATLAB's SVG export hardcodes font-family="Helvetica" regardless of
% what FontName is set to here or what actually rendered the PNG (verified
% empirically -- Helvetica/Arial/Trebuchet MS/Nimbus Sans all produce
% pixel-identical PNGs and the same SVG tag on this system), so this is
% purely documentation of intent for the SVG's declared font-family; it
% has no effect on the PNG or on layout/kerning.
ax.FontName = 'Arial';

plot(ax, Id_OS_norm(os_mask), 1000 * log(10) ./ gm_by_Id_OS_nFET(os_mask), 'LineWidth', 3.5, 'Color', [0 0 1], 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0 0 1]);
plot(ax, Id_Si_norm(si_mask), 1000 * log(10) ./ gm_by_Id_Si_nFET(si_mask), 'LineWidth', 3.5, 'Color', [1 0 0], 'Marker', 's', 'MarkerSize', 8, 'MarkerFaceColor', [1 0 0]);

% MATLAB's SVG export of 'tex'-interpreted subscripts (I_{D}) bakes in a
% fixed subscript offset instead of real glyph-width kerning, so the "D"
% sits with a visibly wider gap after "I" in the SVG than in the PNG
% (where on-screen/raster text layout uses actual font metrics). The
% 'latex' interpreter lays subscripts out properly in both, at the cost
% of falling back to a bold serif (Computer Modern) font for the math
% part instead of the bold DejaVu Sans used elsewhere -- MATLAB's LaTeX
% interpreter doesn't support a sans-serif math font (cmss is rejected).
xlabel(ax, {'I_D [normalized]'}, 'FontWeight', 'bold');
ylabel(ax, {'SS [mV/dec]'}, 'FontWeight', 'bold');

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

exportgraphics(fig2a, fullfile(scriptDir, 'fig2a.pdf'), 'ContentType', 'vector');