import os
import sys
import torch

sys.path.insert(0, os.path.dirname('../vgg16'))

from vgg16.helpers import img_to_tensor

from constants import *

class upload_handler():
    def __init__(self, request, upload_dir, device):
        self.file = request.files[REQ_FIELD]
        self.bp = os.path.dirname(DIR_PNAME)
        self.ud = upload_dir
        self.results = [[]]
        self.device = device

    def _upload_file(self):
        basepath = os.path.dirname(DIR_PNAME)
        file_path = os.path.join(self.bp, self.ud, self.file.filename)
        self.file.save(file_path)
        return file_path

    def _parse_results(self, labels):
        final = {}
        top5 = torch.topk(self.results, NUM_RESULTS)
        accry = [x.item() for x in top5[VALUE_INDEX][VALUE_INDEX]]
        names = [labels[x] for x in top5[NAME_INDEX][VALUE_INDEX]]
        for i, name in enumerate(names):
            final[i+1] = { NAME: name, ACCURACY: accry[i] }
        return final

    def predict(self, vgg16, labels):
        file_path = self._upload_file()
        print(file_path)
        img_tensor = img_to_tensor(file_path, self.device)
        self.results = vgg16(img_tensor)
        return self._parse_results(labels)


