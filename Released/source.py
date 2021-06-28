# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from math import floor, ceil
import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models


class MyDataset(data.Dataset):
    def __init__(self, txt_dir, root, num_frames=40, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(txt_dir, 'r')
        imgs = list()
        for line in fh:
            line = line.rstrip()
            words = line.split()
            vid_name = words[0]
            mos = words[1]
            mos = np.array(mos)
            mos = torch.Tensor(mos)
            imgs.append((vid_name, mos))
        self.num_frames = num_frames
        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        vid_name, label = self.imgs[index]
        IMG = list()
        image = list()
        for i in range(self.num_frames):
            img = Image.open(self.root + vid_name + '_' + str(i+1) + '.png').convert('RGB')
            IMG.append(img)

        if self.transform is not None:
            for j in range(self.num_frames):
                image.append(self.transform(IMG[j]))
        return image, label

    def __len__(self):
        return len(self.imgs)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mini_batch_size = x.shape[0]
        return torch.sum(torch.abs(x - y)) / mini_batch_size


class ANN(nn.Module):
    def __init__(self, input_size=512, reduced_size=256, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        y = self.encoder(x)
        return y


class SpatialPyramidPooling2d(nn.Module):
    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_level):
            level = i + 1
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)

            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'num_level = ' + str(self.num_level) \
            + ', pool_type = ' + str(self.pool_type) + ')'


class MyNet(nn.Module):
    def __init__(self, num_level=3, length=num_frames, iter=3, input_dim=2048, reduced_dim=512, hidden_dim=512,
                 num_frames=num_frames):
        super(MyNet, self).__init__()

        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.length = length
        self.iter = iter
        self.num_level = num_level

        num_grid = 0
        for t in range(self.num_level):
            num_grid += (t + 1) * (t + 1)

        self.backbone = resnet()
        self.ann = ANN(self.input_dim, self.reduced_dim, 1)
        self.gru = nn.GRU(self.reduced_dim, self.hidden_dim, 1, batch_first=True)
        self.p = nn.Linear(num_grid * self.reduced_dim, self.reduced_dim)
        self.q1 = nn.Linear(self.hidden_dim, 128)
        self.q2 = nn.Linear(128, 1)
        self.conv_1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.reduced_dim, kernel_size=1,
                                stride=1, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.spp = SpatialPyramidPooling2d(num_level=self.num_level, pool_type='avg_pool')

        x = list()
        l = list()

        for j in range(self.iter):
            a = 2 ** (j + 1)
            p = 0
            y = list()
            for i in range(0, self.length, a):
                p += 1
                y.append(i)
            x.append(y)
            l.append(p)

        self.scales = l
        self.frames = x

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h0

    def upsample(self, input, factor):
        fix = input.permute(0, 2, 1)
        fix = nn.functional.interpolate(fix, scale_factor=factor, mode='nearest')
        fix = fix.permute(0, 2, 1)
        return fix

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_1(out)
        out = self.spp(out)
        out = torch.squeeze(out)
        out = self.p(out)

        "stage0"
        out = out.view(-1, self.num_frames, self.reduced_dim)
        out_down_1 = out[:, self.frames[0], :]
        out_down_2 = out[:, self.frames[1], :]
        out_down_3 = out[:, self.frames[2], :]

        "stage1"
        outputs_1_0, _ = self.gru(out, self._get_initial_state(out.size(0), out.device))
        outputs_1_1, _ = self.gru(out_down_1, self._get_initial_state(out_down_1.size(0), out.device))
        outputs_1_2, _ = self.gru(out_down_2, self._get_initial_state(out_down_2.size(0), out.device))
        outputs_1_3, _ = self.gru(out_down_3, self._get_initial_state(out_down_3.size(0), out.device))

        "stage2"
        outputs_1_1_fix = self.upsample(outputs_1_1, 2)
        input_2_0 = torch.stack((out, outputs_1_1_fix), 1)
        input_2_0 = torch.mean(input_2_0, 1)
        outputs_2_0, _ = self.gru(input_2_0, self._get_initial_state(input_2_0.size(0), out.device))

        outputs_1_2_fix = self.upsample(outputs_1_2, 2)
        input_2_1 = torch.stack((out_down_1, outputs_1_2_fix), 1)
        input_2_1 = torch.mean(input_2_1, 1)
        outputs_2_1, _ = self.gru(input_2_1, self._get_initial_state(input_2_1.size(0), out.device))

        outputs_1_3_fix = self.upsample(outputs_1_3, 2)
        input_2_2 = torch.stack((out_down_2, outputs_1_3_fix), 1)
        input_2_2 = torch.mean(input_2_2, 1)
        outputs_2_2, _ = self.gru(input_2_2, self._get_initial_state(input_2_2.size(0), out.device))

        "stage3"
        outputs_2_1_fix = self.upsample(outputs_2_1, 2)
        outputs_1_2_fix_2 = self.upsample(outputs_1_2, 4)
        R_3_0 = torch.stack((outputs_2_1_fix, outputs_1_2_fix_2), 1)
        R_3_0 = self.conv_2(R_3_0)
        R_3_0 = torch.squeeze(R_3_0, 1)
        input_3_0 = torch.stack((out, R_3_0), 1)
        input_3_0 = torch.mean(input_3_0, 1)
        outputs_3_0, _ = self.gru(input_3_0, self._get_initial_state(input_3_0.size(0), out.device))

        outputs_2_2_fix = self.upsample(outputs_2_2, 2)
        outputs_1_3_fix_2 = self.upsample(outputs_1_3, 4)
        R_3_1 = torch.stack((outputs_2_2_fix, outputs_1_3_fix_2), 1)
        R_3_1 = self.conv_2(R_3_1)
        R_3_1 = torch.squeeze(R_3_1, 1)
        input_3_1 = torch.stack((out_down_1, R_3_1), 1)
        input_3_1 = torch.mean(input_3_1, 1)
        outputs_3_1, _ = self.gru(input_3_1, self._get_initial_state(input_3_1.size(0), out.device))

        "stage4"
        outputs_3_1_fix = self.upsample(outputs_3_1, 2)
        outputs_2_2_fix_2 = self.upsample(outputs_2_2, 4)
        outputs_1_3_fix_3 = self.upsample(outputs_1_3, 8)
        R_4_0 = torch.stack((outputs_3_1_fix, outputs_2_2_fix_2, outputs_1_3_fix_3), 1)
        R_4_0 = self.conv_3(R_4_0)
        R_4_0 = torch.squeeze(R_4_0, 1)
        input_4_0 = torch.stack((out, R_4_0), 1)
        input_4_0 = torch.mean(input_4_0, 1)
        outputs_4_0, _ = self.gru(input_4_0, self._get_initial_state(input_4_0.size(0), out.device))

        score_1_0 = self.q1(outputs_1_0)
        score_1_0 = self.q2(score_1_0)
        score_1_0 = torch.squeeze(score_1_0, 2)
        score_1_0 = torch.mean(score_1_0, 1)

        score_2_0 = self.q1(outputs_2_0)
        score_2_0 = self.q2(score_2_0)
        score_2_0 = torch.squeeze(score_2_0, 2)
        score_2_0 = torch.mean(score_2_0, 1)

        score_3_0 = self.q1(outputs_3_0)
        score_3_0 = self.q2(score_3_0)
        score_3_0 = torch.squeeze(score_3_0, 2)
        score_3_0 = torch.mean(score_3_0, 1)

        score_4_0 = self.q1(outputs_4_0)
        score_4_0 = self.q2(score_4_0)
        score_4_0 = torch.squeeze(score_4_0, 2)
        score_4_0 = torch.mean(score_4_0, 1)

        return score_4_0, score_3_0, score_2_0, score_1_0


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = ArgumentParser(description='"RIRNet')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_frames', type=int, default=40,
                        help='number of frames from each video (default: 40)')

    parser.add_argument('--database', default='konvid', type=str,
                        help='database name (default: konvid-1k)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()


device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
     ])


train_data = MyDataset(txt_dir='label_path',
                       root='frame_path',
                       transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)

model = MyNet().to(device)

criterion = My_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

total_step = len(train_loader)
curr_lr = args.lr
min_loss = 65532

for epoch in range(args.epochs):
    print(epoch)
    batch_losses = []
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        input = list()

        for q in range(args.batch_size):
            for p in range(args.num_frames):
                input.append(images[p][q])

        pics = torch.stack(input, 0)

        pics = pics.to(device)
        labels = labels.to(device).float()

        outputs = model(pics)

        loss = criterion(outputs[0], labels) \
               + 0.3 * (criterion(outputs[1], labels) + criterion(outputs[2], labels) + criterion(outputs[3], labels))

        batch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            curr_lr /= 5
            update_lr(optimizer, curr_lr)

        print(
            "[Batch %d/%d] [loss: %f]"
            % (i, len(train_loader), loss.item())
        )

    avg_loss = sum(batch_losses) / (len(train_data) // args.batch_size + 1)
    print('Epoch {}, Averaged loss: {:.4f}'.format(epoch+1, avg_loss))

    is_best = avg_loss < min_loss
    min_loss = min(avg_loss, min_loss)

    if is_best:
        torch.save(model.state_dict(), '{}/Epoch{}_loss_{:.4f}.pth'.format('new_checkpoints', epoch, avg_loss))
        print('Save the best weights!')
