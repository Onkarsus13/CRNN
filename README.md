# CRNN
Convolutional Recurrent Neural Network
This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch. Origin software could be found in crnn

Run demo
A demo program can be found in demo.py. Before running the demo, download a pretrained model from Baidu Netdisk or Dropbox. This pretrained model is converted from auther offered one by tool. Put the downloaded model file crnn.pth into directory data/. Then launch the demo by:

python demo.py
The demo reads an example image and recognizes its text content.

Example image: Example Image

Expected output: loading pretrained model from ./data/crnn.pth a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
warp_ctc_pytorch
lmdb
Train a new model
Construct dataset following origin guide. If you want to train with variable length images (keep the origin ratio for example), please modify the tool/create_dataset.py and sort the image according to the text length.
Execute python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda. Explore train.py for details.
