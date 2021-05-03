import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn



def predection(img_path):
    img_path = img_path
    model_path = './data/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    #print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

if __name__ == '__main__':
    p = predection('D:\crnn\data\IMG_20210413_131814__01.jpg')
    print(p)
    