import os

from pathlib import Path

data = Path('./data/six_pm')

for folder in data.iterdir():
    if folder.is_dir():
        not_images = list(filter(lambda item: not item.name.endswith('jpg'), folder.iterdir()))

        if len(not_images) > 0:
            os.remove(str(not_images[0]))

            print(list(not_images))

