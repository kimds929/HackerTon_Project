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

import json

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

from torch.utils.data import DataLoader, TensorDataset

sys.path.append(model_path)
sys.path.append(weight_path)

from KnowledgeTracingModel import KnowledgeTracing_LSTM_MidSchool
from MiddleSchoolKnowledge import kcs_midschool, kcs_to_chapter_midschool, \
                                ch_to_action, action_to_ch, kc_to_idx_all, kc_to_idx
from Generate_InitialTest import Generate_InitialTestSet
from MathSimulation_Agent import MathAgent
from iql_model import IQLTest

# load json
with open(f"{dataset_path}/knowledge_graph_concept.json", "r") as file:
    knowledge_graph = json.load(file)

device = 'cpu'
###############################################################
# kt_model = KnowledgeTracing_LSTM(knowledge_graph)
kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph)

# (load_weights)
load_weights = cPickle.load(open(f"{weight_path}/KnowledgeTracing_LSTM.pkl", 'rb'))
kt_model.load_state_dict(load_weights)
kt_model.to(device)

# RL Model
model_path = f"{model_path}/iql_11-12_model.pth"  # Replace with actual path

iql_model=IQLTest(model_path)



###############################################################

def inference(data):
    # preprocessing
    kcs, ans, dif = data
    kcs_torch = torch.tensor(kcs, dtype=torch.int64).view(-1)
    ans_torch = torch.tensor(ans, dtype=torch.float32).view(-1)
    diff_torch = torch.tensor(dif, dtype=torch.float32).view(-1)
    
    # kt model
    state = kt_model.get_knowledge_state(kcs_torch, ans_torch, diff_torch)
    knowledge_state_df = pd.DataFrame(state.numpy(), columns=['kcs'])
    knowledge_state_df['ch'] = list(kt_model.kcs_to_chapter_midschool.values())*3
    knowledge_state_df['difficulty'] = kt_model.ids_difficulty.view(-1).numpy()
    rl_state = knowledge_state_df.groupby(['difficulty','ch']).mean().sort_index().to_numpy().reshape(-1)

    # rl model
    action = iql_model.infer(rl_state).item()
    ch, diff = kt_model.action_to_ch[action % 102], (action // 102) - 1

    return ch, diff

# (sample data)
kcs = torch.tensor([3,4,3,24])
ans = torch.tensor(np.random.choice([1,-1], size=len(kcs)))
diffs = torch.rand_like(kcs.type(torch.float32))
data = [list(kcs.numpy().astype(float)), list(ans.numpy().astype(float)), list(diffs.numpy().astype(float))]
print(inference(data))









