"""Export the complete Heights matrix, preserving blanks and numeric zero."""
import csv
from pathlib import Path

from .output import new_output


def sample_label(sample):
    suffix = f' [run {sample.native_sample_id}]' if sample.native_sample_id is not None else ''
    return f'{sample.path.stem}{suffix} ({sample.id})'


def write_csv(result, destination, *, checkpoint=lambda: None):
    destination = Path(destination)
    if destination.suffix.lower() != '.csv':
        raise ValueError('Output must have a .csv extension')
    with new_output(destination, checkpoint=checkpoint) as temporary:
        with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            # Filenames can begin with formula characters. The known label
            # prefix makes every sample header literal in spreadsheet readers.
            writer.writerow(['Feature ID', 'm/z', 'RT (min)', 'Polarity'] +
                            [f'{s.id}: {sample_label(s)}' for s in result.samples])
            for i, feature in enumerate(result.features):
                checkpoint()
                writer.writerow([feature.id, feature.mz, feature.rt,
                                 {1: 'positive', -1: 'negative', 0: 'unknown'}[feature.polarity]] +
                                [column[i].height for column in result.measurements])
    return destination
