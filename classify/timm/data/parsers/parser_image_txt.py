
import os

from timm.utils.misc import natural_key

from .parser import Parser
# from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


class ParserImageTxt(Parser):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        self.samples = []
        lines = open(root).read().splitlines()
        for i in lines:
            try:
                info = i.split(';')
                self.samples.append((info[0], int(info[1])))
            except Exception:
                continue
        self.class_to_idx = class_map
        # class_to_idx = None
        # if class_map:
        #     class_to_idx = load_class_map(class_map, root)
        # self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
