import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils.iou
import utils.tools
from tcn import TemporalConvNet
from utils.tools import Timer

# --------------------------------------------------
CUDA_ID = '2'
PROJECT_DIR = '/home/share/huyupeng/release/'
DATASET_DIR_TACOS = f'/home/huyupeng/hdd/dataset/tacos/'
utils.tools.setup_seed(0)
torch.backends.cudnn.deterministic = True
# ---------------------------------------------------------
KERNEL_SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 23]
STRIDE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

DROP_OUT = 0.3
EPOCHS = 1024
LR = 0.0001

LEN_FEATURE_V = 4096
LEN_FEATURE_S = 1024
LEN_FEATURE_B = 128
V = np.sqrt(LEN_FEATURE_B)


class DataSet:
    def __init__(self, t):
        print('dataset init', t)
        self.names = []
        self.feature_s = {}
        self.feature_v = {}
        self.ticks = {}
        self.sim = {}

        videos = pickle.load(open(f'{DATASET_DIR_TACOS}tacos.pkl', 'rb'))
        sentences = pickle.load(open(f'{DATASET_DIR_TACOS}tacos_bert_25_{t}.pkl', 'rb'))
        self.names = list(sentences.keys())
        for name in tqdm(self.names):
            the_feature_v = videos[name]
            sample = 29.4 / 8
            self.feature_v[name] = np.stack([utils.tools.normalize(np.mean(the_feature_v[int(i - sample):int(i)], axis=0)) for i in np.arange(sample, len(the_feature_v) + sample, sample)]).astype(np.float32)
            self.feature_s[name] = sentences[name]['vector'].astype(np.float32)
            self.ticks[name] = []
            len_frame = len(self.feature_v[name])
            for kernel_size, stride in zip(KERNEL_SIZE, STRIDE):
                if kernel_size <= len_frame:
                    self.ticks[name] += map(lambda x: (x - kernel_size, x), range(kernel_size, len_frame + 1, stride))
            self.ticks[name] = np.array(self.ticks[name], dtype=np.float32)
            self.sim[name] = np.zeros([len(self.feature_s[name]), len(self.ticks[name])], dtype=np.float32)
            tick0, tick1 = map(lambda x: np.array(x), zip(*self.ticks[name]))
            for i, timestamps in enumerate(sentences[name]['timestamps']):
                for timestamp in timestamps:
                    timestamp0 = np.array(min(timestamp[0], timestamp[1])) / 29.4
                    timestamp1 = np.array(max(timestamp[0], timestamp[1])) / 29.4
                    self.sim[name][i] = np.maximum(self.sim[name][i], utils.iou.iou(timestamp0, timestamp1, tick0, tick1))
            self.feature_s[name] = torch.from_numpy(self.feature_s[name])
            self.feature_v[name] = torch.from_numpy(self.feature_v[name])
            self.sim[name] = torch.from_numpy(self.sim[name])

    def __getitem__(self, index):
        name = self.names[index]
        res = [
            name,
            self.feature_s[name].cuda(),
            self.feature_v[name].cuda(),
            self.sim[name].cuda()
        ]
        return res

    def __len__(self):
        return len(self.names)


def hash_function(feed):
    feed = func.normalize(feed) * V
    tensor0 = torch.tensor(3.92155, dtype=torch.float32).cuda()
    tensor1 = torch.tensor(1.59362, dtype=torch.float32).cuda()
    res = torch.log(torch.add(torch.pow(feed, 2) * tensor0, 1))
    res = torch.div(res, feed * tensor1 + 0.0000001)
    return res


class VideoModule(nn.Module):
    def __init__(self):
        super(VideoModule, self).__init__()
        self.tcn = TemporalConvNet(LEN_FEATURE_V, [2048, 1024, 512, 256], kernel_size=3, dropout=DROP_OUT)
        self.conv1d = nn.ModuleList([nn.Conv1d(in_channels=256,
                                               out_channels=256,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dilation=1)
                                     for stride, kernel_size in zip(STRIDE, KERNEL_SIZE)])
        self.net = nn.Sequential(
            nn.RReLU(),
            nn.Linear(256, LEN_FEATURE_B),
            nn.RReLU(),
            nn.Linear(LEN_FEATURE_B, LEN_FEATURE_B)
        )

    def forward(self, feed):
        feed = self.tcn(feed)
        out = []
        for kernel_size, conv1d in zip(KERNEL_SIZE, self.conv1d):
            if feed.shape[2] >= kernel_size:
                tmp = conv1d(feed)
                out.append(tmp)
        out = torch.cat(out, dim=2).permute(0, 2, 1)
        out = torch.squeeze(out, dim=0)
        out = self.net(out)
        return out


class SentenceModule(nn.Module):
    def __init__(self):
        super(SentenceModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(LEN_FEATURE_S, 512),
            nn.RReLU(),
            nn.Linear(512, 256),
            nn.RReLU(),
            nn.Linear(256, LEN_FEATURE_B)
        )

    def forward(self, feed):
        out = self.net(feed)
        return out


def train():
    optimizer = torch.optim.Adam([{'params': model_s.parameters()}, {'params': model_v.parameters()}], lr=LR)
    max_performance = 0
    for epoch in range(0, EPOCHS):
        random.shuffle(dataset_train.names)
        for th in range(len(dataset_train) // 2):
            name, feature_s, feature_v, sim = dataset_train[th]
            output_s = hash_function(model_s(feature_s))
            output_v = hash_function(model_v(torch.unsqueeze(feature_v, dim=0).permute(0, 2, 1)))
            loss1 = torch.sum(torch.pow(torch.matmul(output_s, output_v.t()) - sim * LEN_FEATURE_B, 2))
            loss2 = torch.sum(torch.pow(output_s - torch.sign(output_s).detach(), 2)) + torch.sum(torch.pow(output_v - torch.sign(output_v).detach(), 2))
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch)
        res = utils.tools.EvaluateLite.start(dataset_val, model_s, model_v, None, epoch, 1)
        print(res)
        if res[2] > max_performance:
            max_performance = res[2]
            torch.save(model_v.state_dict(), f'{PROJECT_DIR}model/{os.path.basename(__file__)[:-3]}_v.model')
            torch.save(model_s.state_dict(), f'{PROJECT_DIR}model/{os.path.basename(__file__)[:-3]}_s.model')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
    model_s = SentenceModule().cuda()
    model_v = VideoModule().cuda()
    dataset_val = DataSet('val')
    dataset_train = DataSet('train')
    train()
    model_s.load_state_dict(torch.load(f'{PROJECT_DIR}model/{os.path.basename(__file__)[:-3]}_s.model'))
    model_v.load_state_dict(torch.load(f'{PROJECT_DIR}model/{os.path.basename(__file__)[:-3]}_v.model'))
    print('-----------start test-------------')
    dataset_test = DataSet('test')
    utils.tools.Evaluate.start(dataset_test, model_s, model_v)
