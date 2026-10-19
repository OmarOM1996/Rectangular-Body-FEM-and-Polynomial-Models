"""Convert generated FEM NPZ results to MATLAB fitting inputs."""
from pathlib import Path
import numpy as np
from scipy.io import savemat


def main():
    root = Path(__file__).resolve().parent
    inputs = []
    for prefix, folder, count in [('TN', 'FB2_cases', 37), ('VN', 'FB2_validation_cases', 24)]:
        files = sorted((root.parent / folder).glob(f'{prefix}*/*_results.npz'))
        if len(files) != count:
            raise RuntimeError(f'Expected {count} {prefix} results; found {len(files)}. Run FEM first.')
        inputs.extend((prefix, path) for path in files)
    for prefix, path in inputs:
        destination = root / 'data_mat' if prefix == 'TN' else (
            root / 'validation_cases' / path.parent.name / 'FEM_results')
        destination.mkdir(parents=True, exist_ok=True)
        with np.load(path) as archive:
            data = {name: archive[name] for name in archive.files}
        savemat(destination / path.with_suffix('.mat').name, data,
                do_compression=True, long_field_names=True)
    print('Exported 37 fitting and 24 verification cases.')


if __name__ == '__main__':
    main()
