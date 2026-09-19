function fig_to_png(figPath)
%FIG_TO_PNG Save a PNG version of a MATLAB .fig file.
%   FIG_TO_PNG(FIGPATH) opens the figure saved at FIGPATH and exports a
%   .png with the same base name into FIGPATH's parent directory, e.g.
%   fig_to_png('/some/dir/fig1.fig') writes '/some/dir/fig1.png'.

    figPath = char(figPath);
    [parentDir, baseName] = fileparts(figPath);
    pngPath = fullfile(parentDir, [baseName '.png']);

    fig = openfig(figPath, 'invisible');
    % 'Padding','figure' keeps the full figure canvas (including any
    % blank margins the figure was laid out with) -- exportgraphics'
    % default 'tight' padding crops to the bounding box of actually
    % rendered ink, silently discarding deliberate blank space (e.g. a
    % margin reserved via an invisible/transparent placeholder label).
    exportgraphics(fig, pngPath, 'Padding', 'figure');
    close(fig);

    fprintf('Saved %s\n', pngPath);
end
