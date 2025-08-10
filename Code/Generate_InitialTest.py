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

from torch.utils.data import DataLoader, TensorDataset

from KnowledgeTracingModel import KnowledgeTracing_LSTM, KnowledgeTracing_LSTM_MidSchool
# from KnowledgeTracingModel import PeriodicEmbedding, TemporalEmbedding, knowledge_bfs, preprocess_knowledge_bfs,\
#                                 KCsEmbedding, DifficultyEmbedding, AnserEmbedding, KnowledgeTracing_LSTM

# input argument ------------------------------------------------------------------


# load json
with open(f"{dataset_path}/knowledge_graph_concept.json", "r") as file:
    knowledge_graph = json.load(file)

kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph, device='cpu')

# ---------------------------------------------------------------------------------
from scipy.special import softmax

class Generate_InitialTestSet():
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)
        self.all_grade = ['초1', '초2', '초3', '초4', '초5', '초6', '중1', '중2', '중3']
        self.df_concept = pd.read_csv(f"{dataset_path}/Math_Concepts.csv")
        self.knowledge_group_np = cPickle.load(open(f'{dataset_path}/knowledge_group_np.pkl', 'rb'))

    def categorize_difficulty(x):
        if float(x) <= -0.5:
            return -1
        elif float(x) < 0.5:
            return 0
        else:
            return 1
        
    def generate(self, learner_grade, size=10, return_type='ch'):
        df_grade = self.df_concept[self.df_concept['semester'].str.contains(learner_grade)]
        df_grade = df_grade[['id', 'semester']].set_index('id')
        kcs = df_grade.index.to_numpy() # filtered kc의 ID

        is_in_grade = np.isin(self.knowledge_group_np, kcs)
        counts_per_row = np.sum(is_in_grade, axis=1)
        prob_per_kc = softmax(counts_per_row)
        sampled_indices = self.rng.choice(len(counts_per_row), size=size, replace=True, p=prob_per_kc)
        # array([ 5, 63, 20, 63, 20, 20, 63, 63, 63, 51]) sampled_groups
        # 위 KC에서 3학년 문제 중에 uniform sampling
        cold_start_kc = []
        cold_start_diff = []

        for idx in sampled_indices:
            true_indices = np.where(is_in_grade[idx] == True)[0]
            sampled_index = self.rng.choice(true_indices) # true_indices 중에
            knowledge_group_np = np.array(self.knowledge_group_np)
            sampled_kc = knowledge_group_np[idx, sampled_index]
            cold_start_kc.append(sampled_kc)

        if return_type == 'ch':
            cold_start_ch = [kt_model.kcs_to_chapter_midschool[kc] for kc in cold_start_kc]
            return cold_start_ch
        elif return_type == 'kc_ch_pair':
            cold_start_ch = [kt_model.kcs_to_chapter_midschool[kc] for kc in cold_start_kc]
            return list(zip(cold_start_kc, cold_start_ch))
        elif return_type == 'kc':
            return cold_start_kc


# git = Generate_InitialTestSet(random_state=None)
# git.generate('중2', size=20)



