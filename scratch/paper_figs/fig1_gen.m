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
ax.FontSize = min(fig1.Position(3), fig1.Position(4)) / 17.5;

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
hYLabelLeft = ylabel(ax, 'I_{D} [log scale]', 'FontWeight', 'bold', 'Color', 'none');
ylim(ax, [yLow yHigh]);
yticks(ax, [])

yyaxis(ax, 'right'); % Right y-axis (linear scale)
plot(ax, Vg_OS_nFET, Id_OS_nFET/W, 'LineWidth', 3.5, 'Color', [0 0 1], 'LineStyle', '-');
hYLabelRight = ylabel(ax, 'I_{D} [linear scale]', 'FontWeight', 'bold', 'Color', 'none');
yticks(ax, [])

hXLabel = xlabel(ax, 'V_{GS}', 'FontWeight', 'bold', 'Color', 'none');

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

% Force MATLAB to recompute the axes' Position (and therefore where the
% invisible xlabel/ylabels below actually sit) before reading it back --
% without this, their queried Position reflects the pre-LooseInset layout.
drawnow;

% Build the VISIBLE labels now that the invisible xlabel/ylabels above have
% fixed the axes' final Position/margins. Each invisible label's own
% (normalized) Position tells us exactly where MATLAB would have centered
% it, so the hand-built replacement lines up with where the built-in one
% would have gone. The invisible originals are deleted right after, since
% 'Color','none' doesn't reliably stick through the ax.YAxis(:).Color
% assignments below (yyaxis appears to re-link YLabel Color to its ruler).
mainFS = ax.FontSize;
subFS = ax.FontSize * 0.7;
labelGap = 0.01;

set(hXLabel, 'Units', 'normalized');
xLabelPos = get(hXLabel, 'Position');
xLabelAcross = xLabelPos(2);
delete(hXLabel);
place_kerned_label(ax, 0.5, xLabelAcross, 0, ...
    {'V', mainFS; 'GS', subFS}, labelGap, 'bold', 'Helvetica', -1);

set(hYLabelLeft, 'Units', 'normalized');
yLabelLeftPos = get(hYLabelLeft, 'Position');
yLabelLeftAcross = yLabelLeftPos(1);
yLabelLeftRot = get(hYLabelLeft, 'Rotation');
delete(hYLabelLeft);
place_kerned_label(ax, 0.5, yLabelLeftAcross, yLabelLeftRot, ...
    {'I', mainFS; 'D', subFS; ' [log scale]', mainFS}, labelGap, 'bold', 'Helvetica', -1);

set(hYLabelRight, 'Units', 'normalized');
yLabelRightPos = get(hYLabelRight, 'Position');
yLabelRightAcross = yLabelRightPos(1);
yLabelRightRot = get(hYLabelRight, 'Rotation');
delete(hYLabelRight);
place_kerned_label(ax, 0.5, yLabelRightAcross, yLabelRightRot, ...
    {'I', mainFS; 'D', subFS; ' [linear scale]', mainFS}, labelGap, 'bold', 'Helvetica', 1);

% Set the figure and axes background to transparent
set(gcf, 'Color', 'none');
set(ax, 'Color', 'none');

% Save the figure as a MATLAB .fig file in the script directory
savefig(fig1, fullfile(scriptDir, 'fig1.fig'));

function hs = place_kerned_label(ax, alongCenter, acrossFixed, rotationDeg, pieces, gap, fontWeight, fontName, awaySign)
%PLACE_KERNED_LABEL Build a multi-run label with real glyph-width kerning.
%   Places each {string, fontSize} row of PIECES one after another along
%   the direction ROTATIONDEG (degrees, same convention as a text object's
%   'Rotation'), each abutting the previous run's ACTUAL rendered Extent
%   (queried after a drawnow -- text Extent is only meaningful with a real
%   display, so callers must run under Xvfb/a real X session, not
%   -nodisplay) rather than relying on MATLAB's own tex/latex subscript
%   placement (which uses a fixed offset instead of real glyph widths).
%   The whole run is then centered on ALONGCENTER along the reading
%   direction, at the fixed perpendicular offset ACROSSFIXED -- both in
%   the axes' normalized units, matching where a built-in xlabel/ylabel's
%   own Position would put it -- specifically, the edge of the text NEAR
%   the axis (its default alignment leaves the text hanging off that edge
%   away from the axis, e.g. an xlabel's Position is its TOP edge, with
%   the label extending downward/away from there). AWAYSIGN is +1 or -1:
%   the sign, in the across-axis coordinate, of "away from the axes" --
%   used to push the (baseline-anchored) replacement out by its own
%   measured cap-height so its near edge lines up with ACROSSFIXED instead
%   of its baseline sitting there (which would let ascenders poke back
%   over the axis). Only rotationDeg values of 0 or +-90 are supported
%   (the only ones this figure needs): at those angles the reading
%   direction and the fixed perpendicular direction are always exactly
%   axis-aligned, so each can be set as a literal x/y coordinate instead of
%   via a rotated-vector projection (which flips the sign of the
%   perpendicular coordinate at 90 degrees -- fine for a small, near-zero
%   offset, but sends anything with a large offset, like the right-hand
%   y-axis label, far off canvas).
    n = size(pieces, 1);
    hs = gobjects(n, 1);
    alongs = zeros(n, 1);
    isRotated = abs(sind(rotationDeg)) > 0.5;

    for i = 1:n
        hs(i) = text(ax, 0, 0, pieces{i,1}, 'Units', 'normalized', ...
            'FontSize', pieces{i,2}, 'FontWeight', fontWeight, 'FontName', fontName, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'baseline', ...
            'Rotation', rotationDeg);
    end
    drawnow;

    along = 0;
    capHeight = 0;
    for i = 1:n
        alongs(i) = along;
        ext = get(hs(i), 'Extent'); % [x y width height]
        % Extent's (width, height) fields swap which one is the real,
        % content-dependent along-reading-direction size depending on
        % Rotation: at 0 degrees it's width (ext(3)); at 90 it's height
        % (ext(4)) -- the field that looks like "width" at 90 degrees is
        % actually a font-size-only quantity, constant regardless of the
        % string, which silently breaks kerning for any multi-character
        % run once rotated.
        if isRotated
            alongExtent = ext(4);
        else
            alongExtent = ext(3);
        end
        along = along + alongExtent + gap;
        if ~isRotated
            capHeight = max(capHeight, ext(2) + ext(4)); % top edge above baseline
        end
    end
    totalWidth = along - gap;
    shift = alongCenter - totalWidth / 2;
    % The x/y roles in Extent swap along with (width, height) once
    % rotated (see above), so this correction -- derived assuming an
    % unrotated, top-edge-above-baseline reading of ext(2)/ext(4) -- only
    % applies cleanly at rotationDeg == 0; the rotated y-labels already
    % line up correctly without it.
    if ~isRotated
        acrossFixed = acrossFixed + awaySign * capHeight;
    end
    for i = 1:n
        p = alongs(i) + shift;
        if isRotated
            pt = [acrossFixed, p];
        else
            pt = [p, acrossFixed];
        end
        set(hs(i), 'Position', [pt, 0]);
    end
end

