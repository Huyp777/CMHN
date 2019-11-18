import numpy as np
import time
import torch
import os
from tqdm import tqdm
import random


def normalize(feed):
    return feed / (np.linalg.norm(feed) + 1e-7)


def o_inter(s0, t0, s1, t1):
    the_inter = min(t0, t1) - max(s0, s1)
    the_inter = max(0, the_inter)
    return the_inter


def o_union(s0, t0, s1, t1):
    return max(t0, t1) - min(s0, s1)


def o_iou(s0, t0, s1, t1):
    i = o_inter(s0, t0, s1, t1)
    u = o_union(s0, t0, s1, t1)
    if i == 0:
        return 0
    return i / u


def inter(s0, t0, s1, t1):
    the_inter = np.minimum(t0, t1) - np.maximum(s0, s1)
    the_inter = np.maximum(0, the_inter)
    return the_inter


def union(s0, t0, s1, t1):
    return np.maximum(t0, t1) - np.minimum(s0, s1)


def iou(s0, t0, s1, t1):
    i = inter(s0, t0, s1, t1)
    u = union(s0, t0, s1, t1) + 1e-7
    return i / u


class Timer:
    def __init__(self):
        self.sum_time = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            print('start() before end().')
        self.sum_time += time.time() - self.start_time
        self.start_time = None

    def value(self):
        return self.sum_time


class Evaluate:
    @staticmethod
    def nms(indices_all, stick, overlap):
        res = []
        for indices in indices_all:
            new_list = []
            for index in indices:
                flag = True
                for item in new_list:
                    if o_iou(stick[item][0], stick[item][1], stick[index][0], stick[index][1]) > overlap:
                        flag = False
                        break
                if flag is True:
                    new_list.append(index)
                    if len(new_list) == 5:
                        break
            res.append(new_list)
        return res

    @staticmethod
    def get_r_at_iou(sim, indices, rat, threshold):
        cnt = 0
        assert len(sim) == len(indices)
        for sim_item, index in zip(sim, indices):
            for i in range(min(len(index), rat)):
                if sim_item[index[i]] >= threshold:
                    cnt += 1
                    break
        return cnt

    @staticmethod
    def start(dataset_test, model_s, model_v, with_sign=True):
        with torch.no_grad():
            model_s.eval()
            model_v.eval()
            time_all = Timer()
            time_video = Timer()
            time_sentence = Timer()
            iou_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            r_at_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            r_at_5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            cnt_sentences = 0
            m_iou = 0
            for name, feature_s, feature_v, sim in dataset_test:
                ticks = dataset_test.ticks[name]
                feature_v = torch.unsqueeze(feature_v, dim=0).permute(0, 2, 1)
                time_all.start()
                time_video.start()
                output_v = model_v(feature_v)
                if with_sign:
                    output_v = torch.sign(output_v)
                time_video.end()
                time_sentence.start()
                output_s = model_s(feature_s)
                if with_sign:
                    output_s = torch.sign(output_s)
                time_sentence.end()
                time_all.end()
                ans = torch.matmul(output_s, output_v.t())
                _, indices = torch.topk(ans, sim.shape[1])
                indices = indices.cpu().numpy()
                sim = sim.cpu().numpy()
                for the_sim, index in zip(sim, indices):
                    m_iou += the_sim[index[0]]
                for i, the_iou_threshold in enumerate(iou_threshold):
                    indices_ = Evaluate.nms(indices, ticks, the_iou_threshold - 0.05)

                    r_at_1[i] += Evaluate.get_r_at_iou(sim, indices_, 1, the_iou_threshold)
                    r_at_5[i] += Evaluate.get_r_at_iou(sim, indices_, 5, the_iou_threshold)
                cnt_sentences += indices.shape[0]
            print(f'--------------------------------')
            for value in reversed(r_at_1):
                print("%.2f" % (value / cnt_sentences * 100), end=' ')
            for value in reversed(r_at_5):
                print("%.2f" % (value / cnt_sentences * 100), end=' ')
            print("%.7f %.7f" % (
                time_all.value() / cnt_sentences,
                time_all.value()
            ))
            model_s.train()
            model_v.train()


class EvaluateLite:
    @staticmethod
    def start(dataset, model_s, model_v, now_writer, epoch, num, with_sign=True):
        with torch.no_grad():
            iou_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
            r_at_1 = [0, 0, 0, 0, 0]
            cnt_sentences = 0
            model_s.eval()
            model_v.eval()
            for th in range(len(dataset) // num):
                name, feature_s, feature_v, sim = dataset[th]
                output_s = model_s(feature_s)
                output_v = model_v(torch.unsqueeze(feature_v, dim=0).permute(0, 2, 1))
                if with_sign:
                    output_s = torch.sign(output_s)
                    output_v = torch.sign(output_v)
                ans = torch.matmul(output_s, output_v.t())
                _, indices = torch.topk(ans, 1)
                res = torch.gather(sim, dim=1, index=indices)
                for i, the_iou_threshold in enumerate(iou_threshold):
                    r_at_1[i] += torch.sum(torch.ge(res, the_iou_threshold)).item()
                cnt_sentences += indices.shape[0]
            if now_writer is not None:
                for i, the_iou_threshold in enumerate(iou_threshold):
                    now_writer.add_scalar(f'Rat1/IoU={the_iou_threshold}', r_at_1[i] / cnt_sentences * 100, epoch)
            model_s.train()
            model_v.train()
            return np.array(r_at_1) / cnt_sentences * 100


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
