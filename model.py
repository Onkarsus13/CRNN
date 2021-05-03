from typing import Any, List

# from modelplace_api import BaseModel
# from modelplace_api.objects import Device
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

class InferenceModel(BaseModel):

    def predection(self, img_path, model_path):
        self.img_path = img_path
        self.model_path = model_path
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

        self.model = crnn.CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = model.cuda()
        #print('loading pretrained model from %s' % model_path)
        self.model.load_state_dict(torch.load(model_path))

        self.converter = utils.strLabelConverter(alphabet)

        self.transformer = dataset.resizeNormalize((100, 32))
        self.image = Image.open(self.img_path).convert('L')
        self.image = transformer(self.image)
        if torch.cuda.is_available():
            self.image = image.cuda()
        self.image = image.view(1, *image.size())
        self.image = Variable(image)

        model.eval()
        self.preds = model(image)

        _, self.preds = preds.max(2)
        self.preds = preds.transpose(1, 0).contiguous().view(-1)

        self.preds_size = Variable(torch.IntTensor([preds.size(0)]))
        #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        self.sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred