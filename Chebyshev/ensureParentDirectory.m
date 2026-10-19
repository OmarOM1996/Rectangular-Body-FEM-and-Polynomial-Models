function ensureParentDirectory(outputPath)
    parentDirectory = fileparts(outputPath);
    if ~isfolder(parentDirectory)
        mkdir(parentDirectory);
    end
end
