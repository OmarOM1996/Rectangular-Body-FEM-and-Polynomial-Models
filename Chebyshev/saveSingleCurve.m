function saveSingleCurve(x, y, xLabelText, yLabelText, titleText, outputPath, xLimits)
    ensureParentDirectory(outputPath);
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 12, 7]);
    plot(x, real(y), 'LineWidth', 2.0);
    xlabel(xLabelText, 'Interpreter', 'tex');
    ylabel(yLabelText, 'Interpreter', 'tex');
    if strlength(string(titleText)) > 0
        title(titleText, 'Interpreter', 'tex');
    end
    grid on;
    box on;
    if ~isempty(xLimits)
        xlim(xLimits);
    end
    exportgraphics(figureHandle, outputPath, 'Resolution', 220);
    close(figureHandle);
end
