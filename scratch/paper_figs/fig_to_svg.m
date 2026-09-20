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

    % MATLAB stamps font-family="Sans Serif" on every group it emits
    % (axis lines, tick containers, etc.), regardless of the figure's
    % FontName -- harmless when the group has no text in it (as here,
    % since text ends up outlined to paths rather than left live), but it
    % leaves the file declaring the wrong font if anything inspects its
    % markup. Rewrite it to match the FontName actually requested.
    svgText = fileread(svgPath);
    svgText = strrep(svgText, 'font-family="Sans Serif"', 'font-family="Helvetica"');
    fid = fopen(svgPath, 'w');
    fwrite(fid, svgText);
    fclose(fid);

    % Hand-built multi-run labels (see place_kerned_label in fig1_gen.m)
    % come out as live <text>, unlike MATLAB's own xlabel/ylabel/title
    % objects, which get outlined to paths -- live text with an
    % unresolvable FontName renders blank in strict SVG viewers (verified
    % with resvg). Bake those specific runs into path outlines too, so the
    % whole file is equally robust.
    scriptDir = fileparts(mfilename('fullpath'));
    converter = fullfile(scriptDir, 'text_to_paths.py');
    [status, cmdOut] = system(sprintf('/opt/shared_python_science/bin/python3 %s %s', converter, svgPath));
    if status ~= 0
        error('text_to_paths.py failed:\n%s', cmdOut);
    end

    fprintf('Saved %s\n', svgPath);
end
