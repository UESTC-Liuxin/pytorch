# demo.py
import os
import time
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
log_dir = 'logs/test/' + TIMESTAMP
resnet18 = models.resnet18(False)
writer = SummaryWriter(log_dir=log_dir)
writer.flush()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]
# with SummaryWriter() as w:
#     for i in range(5):
#         w.add_hparams({'lr': 0.1*i, 'bsize': i},
#                       {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
for n_iter in range(100):

    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)

    dummy_img = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)

        dummy_audio = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # amplitude of sound should in [-1, 1]
            dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
        writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

        # for name, param in resnet18.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

        # needs tensorboard 0.4RC or later
        writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

root=os.getcwd()
data_root=os.path.join(root,'DATA')
print(data_root)
dataset = datasets.MNIST(root=data_root,train=False, download=False)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
print(images.size())
print(label.size())
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
#
writer.add_image_with_boxes('CoCo',torch.rand(3,128,128),torch.tensor([[10,10,50,50]]))
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
