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
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from six.moves import cPickle
import random



from KnowledgeTracingModel import KnowledgeTracing_LSTM, KnowledgeTracing_LSTM_MidSchool
from Generate_InitialTest import Generate_InitialTestSet

def categorize_difficulty(x):
    if float(x) <= -0.5:
        return -1
    elif float(x) < 0.5:
        return 0
    else:
        return 1

kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%d")
now_time = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")

#------------(Load Dataset)------------
df_concept = pd.read_csv(f"{dataset_path}/Math_Concepts.csv")
with open(f"{dataset_path}/knowledge_graph_concept.json", "r") as file:
            knowledge_graph = json.load(file)
knowledge_group_np = cPickle.load(open(f'{dataset_path}/knowledge_group_np.pkl', 'rb'))
kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph, device='cpu')
#------------(Load Dataset)------------

class RecommendPrior():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.df_concept = df_concept.set_index('id')
        self.knowledge_graph = knowledge_graph
        self.kt_model = kt_model
        
    def give_prior_problem(self, current_kc: str):
        # 버튼을 누를 때마다 current_kc의 '하나의' prior chapter 추천
        prior_kcs = self.knowledge_graph.get(current_kc, {}).get('from', [])
        # prior_concepts = self.df_concept.loc[prior_kcs]
        
        if prior_kcs:
            selected_prior_kc = self.rng.choice(prior_kcs)
        else:
            selected_prior_kc = None
        
        prior_chapter = self.kt_model.kcs_to_chapter_midschool[selected_prior_kc]

        return prior_chapter
    
rp = RecommendPrior()
chap = rp.give_prior_problem('3249')        