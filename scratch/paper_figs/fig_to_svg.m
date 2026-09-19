function fig_to_svg(figPath)
%FIG_TO_SVG Save an SVG version of a MATLAB .fig file, transparent background.
%   FIG_TO_SVG(FIGPATH) opens the figure saved at FIGPATH and exports a
%   .svg with the same base name into FIGPATH's parent directory, e.g.
%   fig_to_svg('/some/dir/fig1.fig') writes '/some/dir/fig1.svg'.

    figPath = char(figPath);
    [parentDir, baseName] = fileparts(figPath);
    svgPath = fullfile(parentDir, [baseName '.svg']);

    fig = openfig(figPath, 'invisible');
    % 'Padding','figure' keeps the full figure canvas (including any
    % blank margins the figure was laid out with) -- exportgraphics'
    % default 'tight' padding crops to the bounding box of actually
    % rendered ink, silently discarding deliberate blank space (e.g. a
    % margin reserved via an invisible/transparent placeholder label).
    % 'BackgroundColor','none' makes the exported canvas colorless
    % (transparent) rather than the default opaque white.
    exportgraphics(fig, svgPath, 'Padding', 'figure', 'BackgroundColor','none');
    close(fig);

    fprintf('Saved %s\n', svgPath);
end
