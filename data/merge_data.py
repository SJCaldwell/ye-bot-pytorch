#!/usr/bin/env python3

from glob import iglob
from pathlib import Path
import os

def combine_dataset(path, genre='rap'):
    all_text = ''
    root_dir = Path(path)
    song_list = [f for f in root_dir.glob('**/*') if f.is_file()]
    for text in song_list:
        with open(text) as song_file:
            all_text += song_file.read()
    with open('all_merged.txt', 'w') as out_file:
        out_file.write(all_text)

if __name__ == "__main__":
   combine_dataset('structure/', genre='rap')

