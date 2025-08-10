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
import torch
import matplotlib.pyplot as plt


from KnowledgeTracingModel import KnowledgeTracing_LSTM, KnowledgeTracing_LSTM_MidSchool
from Generate_InitialTest import Generate_InitialTestSet

kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%d")
now_time = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")

#------------(Load Dataset)------------
df_concept = pd.read_csv(f"{dataset_path}/Math_Concepts.csv")
middle_df_concept = df_concept[df_concept['semester'].str.contains('중1|중2|중3')]
middle_df_concept['chapter_id'] = middle_df_concept['chapter'].apply(lambda x: int(eval(x)['id']))

with open(f"{dataset_path}/knowledge_graph_concept.json", "r") as file:
            knowledge_graph = json.load(file)
knowledge_group_np = cPickle.load(open(f'{dataset_path}/knowledge_group_np.pkl', 'rb'))

kt_model = KnowledgeTracing_LSTM_MidSchool(knowledge_graph, device='cpu')
#------------(Load Dataset)------------

class ShowAnalytics():
    def __init__ (self, student_history, student_grade: str):
        self.student_history = student_history # (KC, Ans, Difficulty) t step까지의 연속된 tuple
        self.student_grade = student_grade
        self.df_concept = df_concept.set_index('id')
        self.knowledge_graph = knowledge_graph
        self.knowledge_group_np = knowledge_group_np
        self.kt_model = kt_model
        self.knowledge_state = self.current_knowledge_state() # 487개 KCs 맞출 확률
        self.kc2chapter = self.kt_model.kcs_to_chapter_midschool

    # df_concept에서 각 학년의 데이터 추출
    def data_perGrade(self):
        perGrade = self.df_concept[self.df_concept['semester'].str.contains(self.student_grade)]
        perGrade['chapter_id'] = perGrade['chapter'].apply(lambda x: int(eval(x)['id']))

        return perGrade
    
    def current_knowledge_state(self): 
        KCs = torch.tensor([item[0] for item in self.student_history]) 
        answer = torch.tensor([item[1] for item in self.student_history])
        difficulty = torch.tensor([item[2] for item in self.student_history]) 
        
        return self.kt_model.get_knowledge_state(KCs, answer, difficulty)
    
    def chapter_prob_dict(self): 
        knowledge_state_np = self.knowledge_state.detach().numpy()
        middle_kcs = middle_df_concept['id'].values
        # 478개의 KCs + 확률
        kc_state_dict = {middle_kcs[i]: knowledge_state_np[i] for i in range(len(middle_kcs))}
        # 102개의 Chapter + 확률
        chapter_prob_dict = {}
        
        for kc, prob in kc_state_dict.items():
            chapter_id = self.kc2chapter[kc]
            
            if chapter_id not in chapter_prob_dict:
                chapter_prob_dict[chapter_id] = []
            
            chapter_prob_dict[chapter_id].append(prob)
        
        averaged_probs = {chapter_id: np.mean(probs) for chapter_id, probs in chapter_prob_dict.items()}
        result_df = pd.DataFrame(list(averaged_probs.items()), columns=['chapter_id', 'averaged_prob'])

        return result_df
    
    # 각 chapter별로 꺽쇠 그래프로 자신의 performance, 자신의 목표 goal
    def plot_performance(self):
        # 해당 학년의 report만 발행
        target_chapter_ids = middle_df_concept[middle_df_concept['semester'].str.contains(self.student_grade)]['chapter_id'].unique()
        result_df = self.chapter_prob_dict()
        filtered_result_df = result_df[result_df['chapter_id'].isin(target_chapter_ids)]
        '''가로축이 chapter_id, 세로축이 평균에서 얼만큼 벗어났는지'''
        
        
    def plot_struggling_kcs(self):
        struggling_kcs = self.get_struggling_kcs()
        
    def generate_report(self):
        print(f"Generating analytics report for student {self.student_id}")
        
# student_history = [
#     (1, 1, 3),  # (KC: 1, Answer: Correct (1), Difficulty: 3)
#     (2, 0, 2)   # (KC: 2, Answer: Incorrect (0), Difficulty: 2)
# ]

# anal = ShowAnalytics
# anal.plot_performance()

