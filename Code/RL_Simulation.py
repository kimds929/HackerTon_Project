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
# from Analytics import ShowAnalytics

# load json
with open(f"{base_path}/dataset/knowledge_graph_concept.json", "r") as file:
    knowledge_graph = json.load(file)

# kt_model = KnowledgeTracing_LSTM(knowledge_graph)
kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph)
# (load_weights)
load_weights = cPickle.load(open(f"{weight_path}/KnowledgeTracing_LSTM.pkl", 'rb'))
kt_model.load_state_dict(load_weights)
# print('load_weights')

# RL Model
model_path = f"{model_path}/iql_11-12_model.pth"  # Replace with actual path
iql_model=IQLTest(model_path)


df_concept_sort = pd.read_csv(f"{dataset_path}/Math_Concepts_sort.csv")
df_concept_sort[df_concept_sort['id'].astype(int) == 3804]
df_concept_sort[df_concept_sort['id'].astype(int) == 10609]

df_concept_sort['id'].to_list()


from tqdm.auto import tqdm

git = Generate_InitialTestSet()

device = 'cpu'
# device = 'cuda:0'
aa = MathAgent(kt_model, grade='중2')
aa.get_knowledge_state()

diff_mean = 0
diff_std = 1
n_iter = 50
n_init_test = 15
n_of_testset = 5

knowledge_hist = [aa.get_knowledge_state().mean()]
pbar = tqdm(total=n_iter, desc="Processing")
state = None


len(list(kt_model.kcs_to_chapter_midschool.values())*3)

i = 0
while(i<n_iter):
    pbar.set_description(f"{i}/{n_iter} iter")
    if i == 0:
        kc = git.generate(aa.grade, size=n_init_test, return_type='kc')
        diff = np.random.normal(diff_mean, diff_std, size=len(kc))
        i += n_init_test-1
        pbar.update(n_init_test-1)
    else:
        knowledge_state_df = pd.DataFrame(state.numpy(), columns=['kcs'])
        knowledge_state_df['ch'] = list(kt_model.kcs_to_chapter_midschool.values())*3
        knowledge_state_df['difficulty'] = kt_model.ids_difficulty.view(-1).numpy()
        rl_state = knowledge_state_df.groupby(['difficulty','ch']).mean().sort_index().to_numpy().reshape(-1)
        # rl model
        action = iql_model.infer(rl_state).item()

        ch, diff_rl = kt_model.action_to_ch[action % 102], (action // 102) - 1
        kc = list(np.random.choice([k for k, v in kcs_to_chapter_midschool.items() if v == ch], size=5))
        diff = [diff_rl]*n_of_testset
        
        # kc = []
        # diff = []
        # for _ in range(n_of_testset):
        #     ch, diff_rl = kt_model.action_to_ch[action % 102], (action // 102) - 1
        #     ch_to_kc = np.random.choice([k for k, v in kcs_to_chapter_midschool.items() if v == ch])
            
        #     np.random.choice([k for k, v in kcs_to_chapter_midschool.items() if v == ch])
        #     kc.append(ch_to_kc)
        #     diff.append(diff_rl)
        print(diff)
        i += n_of_testset-1
        pbar.update(n_of_testset-1)

    aa.solve_simulation(kc, diff)
    state = aa.get_knowledge_state(return_all=True)
    
    knowledge_hist.append(state[aa.grade_idx].mean())

    i += 1
    pbar.set_postfix(knowledge_level=f"{state[aa.grade_idx].mean().item():.2f}")
    pbar.update(1)
    
pbar.close()

plt.plot(knowledge_hist, '-', color='orange', alpha=0.5)

# cPickle.dump([k.item() for k in knowledge_hist], open(f'{dataset_path}/rl_policy_4.pkl', 'wb'))



knowledge_hist_rl_list = []
knowledge_hist_rl_list.append(cPickle.load(open(f'{dataset_path}/rl_policy_1.pkl', 'rb')))
knowledge_hist_rl_list.append(cPickle.load(open(f'{dataset_path}/rl_policy_2.pkl', 'rb')))
knowledge_hist_rl_list.append(cPickle.load(open(f'{dataset_path}/rl_policy_3.pkl', 'rb')))
# knowledge_hist_rl_list.append(cPickle.load(open(f'{dataset_path}/rl_policy_4.pkl', 'rb')))
knowledge_hist_rl_list.append(knowledge_hist)

knowledge_hist_list = []
knowledge_hist_list.append(cPickle.load(open(f'{dataset_path}/agent_random_policy1(mean_-1,std_05).pkl', 'rb')))
# knowledge_hist_list.append(cPickle.load(open(f'{dataset_path}/agent_random_policy1(mean_-1,std_05)2.pkl', 'rb')))
knowledge_hist_list.append(cPickle.load(open(f'{dataset_path}/agent_random_policy1(mean_0,std_1)2.pkl', 'rb')))
knowledge_hist_list.append(cPickle.load(open(f'{dataset_path}/agent_random_policy1(mean_1,std_05)2.pkl', 'rb')))
knowledge_hist_list.append(cPickle.load(open(f'{dataset_path}/agent_random_policy1(mean_1,std_05)3.pkl', 'rb')))




for ei, kh in enumerate(knowledge_hist_rl_list):
    if ei == 0:
        plt.plot(np.arange(len(kh))*6.2, kh, '-', color='steelblue', alpha=0.5, label='rl_policy')
    else:
        plt.plot(np.arange(len(kh))*6.2, kh, '-', color='steelblue', alpha=0.5)

for ei, kh in enumerate(knowledge_hist_list):
    if ei == 0:
        plt.plot(np.arange(len(kh))*1.37, kh, '-', color='orange', alpha=0.5, label='random_policy')
    else:
        plt.plot(np.arange(len(kh))*1.37, kh, '-', color='orange', alpha=0.5)
plt.ylim(0.4,1)
plt.legend(loc='upper left')
plt.xlabel('problem')
plt.ylabel('avg_knowlege_level')
plt.show()



# kcs = torch.tensor([3,4,3,24])
# ans = torch.tensor(np.random.choice([-1,1], size=len(kcs))).type(torch.float32)
# diffs = torch.rand_like(kcs.type(torch.float32))
# state = kt_model.get_knowledge_state(kcs,ans, diffs)

# knowledge_state_df = pd.DataFrame(state.numpy(), columns=['kcs'])
# knowledge_state_df['ch'] = list(kt_model.kcs_to_chapter_midschool.values())*3
# knowledge_state_df['difficulty'] = kt_model.ids_difficulty.view(-1).numpy()
# rl_state = knowledge_state_df.groupby(['difficulty','ch']).mean().sort_index().to_numpy().reshape(-1)











# # np.random.choicekcs_midschool
# seq_size = np.random.randint(10,20)

# kcs = torch.tensor(np.random.choice(kcs_midschool, size=seq_size), dtype=torch.int64)
# # kcs = torch.tensor(3)
# ans = torch.tensor(np.random.choice([-1,1], size=kcs.shape), dtype=torch.float32)
# diff = torch.rand_like(kcs.type(torch.float32))

# print(diff)
# st = kt_model.get_knowledge_state(kcs, ans, diff).view(3,-1)
# print(st.min(dim=-1), st.max(dim=-1))
# # plt.imshow(kt_model.get_knowledge_state().view(-1,2))
# # plt.colorbar()








