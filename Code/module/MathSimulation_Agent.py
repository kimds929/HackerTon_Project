import os
file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.replace('\\','/').split('/')[:[i for i, d in enumerate(file_path.replace('\\','/').split('/')) if 'MathProject' in d][0]+1])

import sys
# base_path = r'/home/kimds929/MathProject'
dataset_path = f"{base_path}/dataset"
model_path = f"{base_path}/model"
module_path = f"{base_path}/module"
weight_path = f"{base_path}/weight"

sys.path.append(base_path)
sys.path.append(model_path)
sys.path.append(module_path)
sys.path.append(weight_path)

import pytz
from datetime import datetime, timedelta
kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%d")
now_time = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")


import json
from six.moves import cPickle
from tqdm.auto import tqdm
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import json
from torch.utils.data import DataLoader, TensorDataset

from KnowledgeTracingModel import KnowledgeTracing_LSTM_MidSchool
from MiddleSchoolKnowledge import kcs_midschool, kcs_to_chapter_midschool, \
                                ch_to_action, action_to_ch, kc_to_idx_all, kc_to_idx
from Generate_InitialTest import Generate_InitialTestSet



# load json
with open(f"{dataset_path}/knowledge_graph_concept.json", "r") as file:
    knowledge_graph = json.load(file)

kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph)
# (load_weights)
load_weights = cPickle.load(open(f"{weight_path}/KnowledgeTracing_LSTM.pkl", 'rb'))
kt_model.load_state_dict(load_weights)
# print('load_weights')

# kt_model.predict(kcs, ans, diff)

class MathAgent():
    def __init__(self, kt_model, grade=None, random_state=None):
        self.grade = grade
        self.history_kcs = []
        self.history_ans = []
        self.history_diff = []

        self.knowlege_model = kt_model
        self.rng = np.random.RandomState(random_state)
        self.sigma = abs(self.rng.normal(0.03,0.01))

        self.kc_to_idx = kc_to_idx[grade]
        self.grade_idx = torch.tensor(list(self.kc_to_idx.values())).type(torch.int64)

        self.device=kt_model.device

    def predict(self, kcs, diff):
        kcs_torch = torch.tensor(kcs).type(torch.int64).view(-1,1).to(self.device)
        ans_torch = torch.zeros_like(kcs_torch).type(torch.float32).to(self.device)
        diff_torch = torch.tensor(diff).type(torch.float32).view(-1,1).to(self.device)

        if len(self.history_kcs) == 0:
            kcs_torch_cat = kcs_torch
            ans_torch_cat = ans_torch
            diff_torch_cat = diff_torch
        else:
            hist_kcs_torch = torch.tensor(self.history_kcs).type(torch.int64).view(-1).to(self.device)
            hist_ans_torch = torch.tensor(self.history_ans).type(torch.float32).view(-1).to(self.device)
            hist_diff_torch = torch.tensor(self.history_diff).type(torch.float32).view(-1).to(self.device)

            hist_kcs_expand = hist_kcs_torch.expand(kcs_torch.shape[0], hist_kcs_torch.shape[-1])
            hist_ans_expand = hist_ans_torch.expand(ans_torch.shape[0], hist_ans_torch.shape[-1])
            hist_diff_expand = hist_diff_torch.expand(diff_torch.shape[0], hist_diff_torch.shape[-1])

            kcs_torch_cat = torch.cat([hist_kcs_expand, kcs_torch], dim=-1)
            ans_torch_cat = torch.cat([hist_ans_expand, ans_torch], dim=-1)
            diff_torch_cat = torch.cat([hist_diff_expand, diff_torch], dim=-1)
        
        with torch.no_grad():
            pred_prob = self.knowlege_model.predict(kcs_torch_cat, ans_torch_cat, diff_torch_cat)[:,-1:].view(-1)
            return pred_prob

    def add_history(self, kcs, diff, ans):
        kcs_torch = torch.tensor(kcs).type(torch.int64).view(-1,1).to(self.device)
        ans_observe_torch = torch.tensor(ans).type(torch.float32).view(-1,1).to(self.device)
        diff_torch = torch.tensor(diff).type(torch.float32).view(-1,1).to(self.device)
        
        self.history_kcs = self.history_kcs + list(kcs_torch.view(-1).detach().to('cpu').numpy())
        self.history_ans = self.history_ans + list(ans_observe_torch.view(-1).detach().to('cpu').numpy())
        self.history_diff = self.history_diff + list(diff_torch.view(-1).detach().to('cpu').numpy())

    def solve_simulation(self, kcs, diff, save_hist=True):
        kcs_torch = torch.tensor(kcs).type(torch.int64).view(-1,1).to(self.device)
        ans_torch = torch.zeros_like(kcs_torch).type(torch.float32).to(self.device)
        diff_torch = torch.tensor(diff).type(torch.float32).view(-1,1).to(self.device)
        
        pred_prob = self.predict(kcs, diff)     # predict
        perturbation = torch.normal(0, self.sigma, size=pred_prob.shape).to(self.device)

        pred_with_perturb = pred_prob + perturbation
        observe_pred = torch.clamp(pred_with_perturb, min=0.0, max=1.0)
        observe_ans = torch.tensor(np.random.binomial(1, observe_pred.detach().to('cpu').numpy(), size=len(observe_pred))).type(torch.float32).view(-1,1)*2 -1
        observe_ans = observe_ans.to(self.device)
        if save_hist is True:
            self.history_kcs = self.history_kcs + list(kcs_torch.view(-1).detach().to('cpu').numpy())
            self.history_ans = self.history_ans + list(observe_ans.view(-1).detach().to('cpu').numpy())
            self.history_diff = self.history_diff + list(diff_torch.view(-1).detach().to('cpu').numpy())
        kcs_torch = kcs_torch.detach().to('cpu')
        observe_ans = observe_ans.detach().to('cpu')
        diff_torch = diff_torch.detach().to('cpu')
        return observe_ans
        # return (kcs_torch, observe_ans, diff_torch)

    def get_knowledge_state(self, return_all=False):
        if len(self.history_kcs) == 0:
            knowledge_state = self.knowlege_model.get_knowledge_state()
        else:
            hist_kcs_torch = torch.tensor(self.history_kcs).type(torch.int64).view(-1).to(self.device)
            hist_ans_torch = torch.tensor(self.history_ans).type(torch.float32).view(-1).to(self.device)
            hist_diff_torch = torch.tensor(self.history_diff).type(torch.float32).view(-1).to(self.device)

            knowledge_state = self.knowlege_model.get_knowledge_state(hist_kcs_torch, hist_ans_torch, hist_diff_torch)

        if return_all:
            return knowledge_state
        else:
            return knowledge_state[self.grade_idx]





# git = Generate_InitialTestSet()

# # (device)
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

# device = 'cuda:0'
# kt_model.to(device)


# aa = MathAgent(kt_model, grade='중2')


# aa = MathAgent(grade='중2', device='cpu')

# st = aa.get_knowledge_state()
# print(st.min(), st.max())

# kc_init = git.generate(aa.grade, size=10, return_type='kc')
# diff_init = np.random.normal(0, 1, size=len(kc_init))
# print(kc_init, diff_init)
# aa.solve_simulation(kc_init, diff_init)

# kcs = torch.tensor([3,4,3,24])
# diffs = torch.rand_like(kcs.type(torch.float32))
# aa.solve_simulation(kcs, diffs)


# aa.history_kcs
# aa.history_ans
# aa.history_diff




