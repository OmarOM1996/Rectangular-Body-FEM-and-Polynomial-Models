function saveSummary(x, addedMassStar, dampingStar, xLabelText, ...
    titleText, outputPath, xLimits)

    ensureParentDirectory(outputPath);
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 18, 9]);
    layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    panels = {
        1, 1, 'A11^*', 'B11^*', 'Surge: A11 and B11';
        2, 2, 'A33^*', 'B33^*', 'Heave: A33 and B33';
        1, 3, 'A15^*', 'B15^*', 'Surge-pitch: A15 and B15';
        3, 3, 'A55^*', 'B55^*', 'Pitch: A55 and B55'
    };

    for panelIndex = 1:size(panels, 1)
        row = panels{panelIndex, 1};
        column = panels{panelIndex, 2};
        nexttile;
        plot(x, squeeze(addedMassStar(:, row, column)), ...
            'LineWidth', 2.0, 'DisplayName', panels{panelIndex, 3});
        hold on;
        plot(x, squeeze(dampingStar(:, row, column)), ...
            'LineWidth', 2.0, 'DisplayName', panels{panelIndex, 4});
        hold off;
        xlabel(xLabelText, 'Interpreter', 'tex');
        ylabel('Nondimensional coefficient [-]');
        title(panels{panelIndex, 5});
        grid on;
        box on;
        legend('Location', 'best');
        if ~isempty(xLimits)
            xlim(xLimits);
        end
    end

    exportgraphics(figureHandle, outputPath, 'Resolution', 220);
    close(figureHandle);
end
