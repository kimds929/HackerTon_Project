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

from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, TensorDataset
from MiddleSchoolKnowledge import kcs_midschool, kcs_to_chapter_midschool, ch_to_action, action_to_ch, kc_to_idx_all, kc_to_idx

# ---------------------------------------------------------------------------
class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Linear Component
        self.linear_layer = nn.Linear(input_dim , 1)
        if embed_dim % 2 == 0:
            self.linear_layer2 = nn.Linear(input_dim , 1)
        else:
            self.linear_layer2 = None
        
        # Periodic Components
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.periodic_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

        # NonLinear Purse Periodic Component
        self.nonlinear_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.nonlinear_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

    def forward(self, x):
        # Linear Component
        linear_term = self.linear_layer(x)
        
        # Periodic Component
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)

        # NonLinear Purse Periodic Component
        nonlinear_term = torch.sign(torch.sin(x @ self.nonlinear_weights + self.nonlinear_bias))
        
        # Combine All Components
        if self.linear_layer2 is None:
            return torch.cat([linear_term, periodic_term, nonlinear_term], dim=-1)
        else:
            linear_term2 = self.linear_layer2(x)
            return torch.cat([linear_term, linear_term2, periodic_term, nonlinear_term], dim=-1)

# TemporalEmbedding
class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = input_dim * embed_dim

        if hidden_dim is None:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=embed_dim) for _ in range(input_dim)])
        else:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=hidden_dim) for _ in range(input_dim)])
            self.hidden_layer = nn.Linear(input_dim*hidden_dim, embed_dim)
            self.embed_dim = embed_dim
    
    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise Exception(f"input shape does not match.")
        # emb_outputs = [layer(x[:,i:i+1]) for i, layer in enumerate(self.temporal_embed_layers)]
        emb_outputs = [layer(x[...,i:i+1]) for i, layer in enumerate(self.temporal_embed_layers)]
        output = torch.cat(emb_outputs, dim=1)
        if self.hidden_dim is not None:
            output = self.hidden_layer(output)

        return output

# -------------------------------------------------------------------------------------------





from collections import deque, defaultdict

def knowledge_bfs(graph, start, from_to='from'):
    visited = set()  # 방문한 노드를 추적하기 위한 집합
    queue = deque([(start, 0)])  # 큐에 (노드, 깊이) 튜플을 추가
    visited.add(start)

    depth_nodes = defaultdict(list)  # 깊이별 노드를 저장할 딕셔너리

    while queue:
        node, depth = queue.popleft()  # 노드와 깊이를 꺼냄
        depth_nodes[depth].append(node)  # 현재 깊이의 노드로 추가

        # 현재 노드의 인접 노드들에 대해 탐색
        for neighbor in graph[str(node)][from_to]:
            if neighbor not in visited:  # 방문하지 않은 노드만 큐에 추가
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))  # 깊이를 증가시켜 큐에 추가

    # return depth_nodes
    return {k: np.array(v) for k, v in depth_nodes.items()}

# df_concept_idx.loc[7941]
# df_concept_idx.loc[8122]
# df_concept_idx.loc[447]

# knowledge_bfs(knowledge_graph, 8122, from_to='from')
# knowledge_bfs(knowledge_graph, 8122, from_to='to')
# knowledge_bfs(knowledge_graph, 9219, from_to='from')
# knowledge_bfs(knowledge_graph, 9219, from_to='to')


# preprocess_knowledge_bfs : 모든 노드의 BFS를 미리 계산하고 torch tensor형태로 저장
def preprocess_knowledge_bfs(graph, node_size=16, depth_size=10, capa_size=8, from_to='from'):

    # node_depth_map = torch.ones([node_size, depth_size, capa_size]).type(torch.int64) * (-1)
    node_depth_map = torch.zeros([node_size, depth_size, capa_size]).type(torch.int64)

    for node in range(node_size):
        if str(node) in  graph.keys():
            visited = set()
            cur_depth = 0
            cur_depth_nodes = []
            queue = deque([(node, cur_depth)])
            visited.add(node)

            while queue:
                current_node, depth = queue.popleft()

                if cur_depth < depth:
                    node_depth_map[node][cur_depth][:len(cur_depth_nodes)] = torch.tensor(cur_depth_nodes, dtype=torch.int64)
                    cur_depth += 1
                    cur_depth_nodes = []
                
                cur_depth_nodes.append(current_node)

                for neighbor in graph[str(current_node)][from_to]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            if len(cur_depth_nodes) > 0:
                node_depth_map[node][cur_depth][:len(cur_depth_nodes)] = torch.tensor(cur_depth_nodes, dtype=torch.int64)

    return node_depth_map

# -------------------------------------------------------------------------------------------


# KCsEmbedding
class KCsEmbedding(nn.Module):
    def __init__(self, knowledge_graph, KCs_size=12000, embed_dim=16, 
                 preKnowledgeWeight=1, postKnowledgeWeight=0.3,
                 preKnowledgeDepth=40, preKnowledgeCapa=150,
                 postKnowledgeDepth=40, postKnowledgeCapa=150,
                 ):
        super().__init__()

        self.embed_dim = embed_dim

        self.knowledge_graph = knowledge_graph
        self.pre_graph_embeddings = preprocess_knowledge_bfs(knowledge_graph, node_size=KCs_size, depth_size=preKnowledgeDepth, capa_size=preKnowledgeCapa, from_to='from')
        self.post_graph_embeddings = preprocess_knowledge_bfs(knowledge_graph, node_size=KCs_size, depth_size=postKnowledgeDepth, capa_size=postKnowledgeCapa, from_to='to')

        self.preKnowledgeWeight = preKnowledgeWeight
        self.postKnowledgeWeight = postKnowledgeWeight

        self.KCs_embedding = nn.Embedding(num_embeddings=KCs_size, embedding_dim=embed_dim)
        nn.init.constant_(self.KCs_embedding.weight, 0.5)
        # nn.init.constant_(self.KCs_embedding.weight[0], 0)
    
    def zero_initialize_first_embedding(self):
        nn.init.constant_(self.KCs_embedding.weight[0], 0)

    def knowledge_embedding(self, x, graph_embedding, weight):
        # 0-th embedding zero initialize
        self.zero_initialize_first_embedding()

        # embedding
        graph_embedding = graph_embedding.to(x.device)
        weights = torch.tensor([i**weight for i in range(graph_embedding.shape[1])], dtype=torch.float32).to(x.device)

        # dim
        for _ in range(x.ndim):
            weights = weights.unsqueeze(0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        graph_x = graph_embedding[x]
        graph_embed_x = self.KCs_embedding(graph_x)
        weighted_graph_embed_x = weights * graph_embed_x
        weighted_cumsum_graph_embed_x = torch.mean(weighted_graph_embed_x[..., 1:,:,:], dim=(-3, -2))
        return weighted_cumsum_graph_embed_x

    def forward(self, x):
        embed = self.KCs_embedding(x)

        output = embed
        if self.knowledge_graph is not None:
            # preKnowlede
            preKnowlede_emged = self.knowledge_embedding(x, graph_embedding=self.pre_graph_embeddings, weight=self.preKnowledgeWeight)
            postKnowlede_embed = self.knowledge_embedding(x, graph_embedding=self.post_graph_embeddings, weight=self.postKnowledgeWeight)

            output += preKnowlede_emged + postKnowlede_embed
        return output

# a = torch.rand(40,150,16)

# ke = KCsEmbedding(knowledge_graph)
# ke(torch.tensor([[3,4,3],[10,24,5]]))
# ke(torch.tensor([3,4,3]))

# -------------------------------------------------------------------------------------------
# DifficultyEmbedding
class DifficultyEmbedding(nn.Module):
    def __init__(self, input_dim=1, embed_dim=4, hidden_dim=16):
        super().__init__()
        self.periodic_embedding = TemporalEmbedding(input_dim=input_dim, embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.unsqueeze(-1)
        x_ndim = x.ndim
        if x_ndim < 2:
            for _ in range(2-x_ndim):
                x = x.unsqueeze(0)
        
        output = self.periodic_embedding(x)

        if x_ndim < 2:
            for _ in range(2-x_ndim):
                output = output.squeeze(0)
        return output


# de = DifficultyEmbedding()
# de(torch.rand(2,3))       # (batch, seq, embedding)
# -------------------------------------------------------------------------------------------



class AnserEmbedding(nn.Module):
    def __init__(self, hidden_dim=16, embed_dim=8):
        super().__init__()
        self.ans_layer = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim, bias=False)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, embed_dim, bias=False)
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.ans_layer(x)

# ae = AnserEmbedding()
# x1 =  torch.tensor([1,0,-1], dtype=torch.float32).view(-1,1)
# x2 =  torch.tensor([[1,0,0],[0,0,0],[1,1,-1]], dtype=torch.float32)
# x1.shape
# ae(x1)
# ae(x2)

########################################################################################
# KnowledgeTracing
class KnowledgeTracing_LSTM(nn.Module):
    def __init__(self, knowledge_graph, KCs_size=12000,
                 KCs_embed_dim = 32, difficulty_embed_dim=8, ans_embed_dim = 8,

                 preKnowledgeWeight=0.7, postKnowledgeWeight=0.2,
                 preKnowledgeDepth=40, preKnowledgeCapa=150,
                 postKnowledgeDepth=40, postKnowledgeCapa=150,
                 rnn_layers=3, hidden_dim=64, output_dim=1, device='cpu'):
        super().__init__()
        self.preKnowledgeWeight = preKnowledgeWeight
        self.postKnowledgeWeight = postKnowledgeWeight
        self.output_dim = output_dim
        self.knowledge_graph = knowledge_graph
        self.knowledge_ids = sorted(list(set(reduce(lambda x, y: x+y, [[int(k)] + v['from'] + v['to'] for k, v in knowledge_graph.items()]))))
        self.ids_KCs = torch.tensor(self.knowledge_ids).tile(3).type(torch.int64).view(-1,1)
        self.ids_answer = torch.zeros_like(self.ids_KCs).type(torch.float32)
        self.ids_difficulty = (torch.tensor([-1,0,1]).view(-1,1) * torch.ones_like(torch.tensor(self.knowledge_ids))).type(torch.float32).view(-1,1)
        self.init_knowledge_state = None
        self.hist_knowledge_state = {}
        self.device = device

        # embedding
        self.KCs_graph_embedding = KCsEmbedding(KCs_size=KCs_size, embed_dim=KCs_embed_dim, knowledge_graph=knowledge_graph, 
                preKnowledgeWeight=preKnowledgeWeight, postKnowledgeWeight=postKnowledgeWeight,
                preKnowledgeDepth=preKnowledgeDepth, preKnowledgeCapa=preKnowledgeCapa, postKnowledgeDepth=postKnowledgeDepth, postKnowledgeCapa=postKnowledgeCapa)
        
        self.AnserEmbedding = AnserEmbedding(embed_dim=ans_embed_dim, hidden_dim=ans_embed_dim*2)

        self.difficulty_embedding = DifficultyEmbedding(input_dim=1, embed_dim=difficulty_embed_dim, hidden_dim=difficulty_embed_dim*2)

        self.rnn_embed_dim = self.KCs_graph_embedding.embed_dim + self.AnserEmbedding.embed_dim + self.difficulty_embedding.embed_dim

        self.rnn_layer = nn.LSTM(input_size= self.rnn_embed_dim,
                     hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True, bias=True,
                     dropout=0,)

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim, 16)
            ,nn.ReLU()
            ,nn.Linear(16, 4)
            ,nn.ReLU()
            ,nn.Linear(4, output_dim)
        )

    def forward(self, KCs, answer, difficulty):
        with torch.no_grad():
            self.device = KCs.device
        
        KCs_embed = self.KCs_graph_embedding(KCs)       # (batch, seq, KCs_embedding)
        ans_embed = self.difficulty_embedding(answer)    # (batch, seq, ans_embedding)
        difficulty_embed = self.difficulty_embedding(difficulty)    # (batch, seq, difficulty_embedding)

        embedding = torch.cat([KCs_embed, ans_embed, difficulty_embed], dim=-1)    # (batch, seq, KCs + difficulty embedding)

        # RNN Input Dimension에 맞추기 위함
        emb_ndim = embedding.ndim
        if emb_ndim < 3:
            for _ in range(3-emb_ndim):
                embedding = embedding.unsqueeze(0)

        # lstm
        hidden_out, (last_out, cell_out) = self.rnn_layer(embedding)
        # hidden_out.shape      # hidden_out: 각 타임스텝에서 LSTM이 생성한 출력 (batch, seq, output_dim)
        # last_out.shape        # last_out: 마지막 타임스텝에서의 hidden state 값 (num_layers, batch, output_dim)
        # cell_out.shape        # cell_out: 마지막 타임스텝에서의 cell state 값 (num_layers, batch, output_dim)

        # fc layer for prediction
        fc_out = self.fc_layer(hidden_out)

        # RNN Output을 Input Dimension에 맞추기 위함
        output = fc_out
        if emb_ndim < 3:
            for _ in range(3-emb_ndim):
                output = output.squeeze(0)
        
        if self.output_dim == 1:
            return output.squeeze(-1)
        else:
            return tuple([t.squeeze(-1) for t in torch.chunk(output, self.output_dim, dim=-1)])

    def predict(self, KCs, answer, difficulty):
        with torch.no_grad():
            output = self.forward(KCs, answer, difficulty)
            if type(output) == tuple:
                return tuple([(torch.tanh(t)/2+0.5) for t in output])
            else:
                return torch.tanh(output)/2 + 0.5
    
    def get_knowledge_state(self, KCs=None, answer=None, difficulty=None, batch_size=64):
        if KCs is None and answer is None and difficulty is None:
            self.init_knowledge_state = self.predict(self.ids_KCs.to(self.device), self.ids_answer.to(self.device), self.ids_difficulty.to(self.device))
            return self.init_knowledge_state
        else:
            if KCs.shape == answer.shape == difficulty.shape and (KCs.ndim * answer.ndim*difficulty.ndim)==1:
                kcs_broadcast = KCs.expand(self.ids_KCs.shape[0], KCs.shape[-1])
                answer_broadcast = answer.expand(self.ids_answer.shape[0], answer.shape[-1])
                difficulty_broadcast = difficulty.expand(self.ids_difficulty.shape[0], difficulty.shape[-1])

                kcs_cat = torch.cat([kcs_broadcast, self.ids_KCs], dim=-1)
                answer_cat = torch.cat([answer_broadcast, self.ids_answer], dim=-1)
                difficulty_cat = torch.cat([difficulty_broadcast, self.ids_difficulty], dim=-1)

                batch_size = batch_size
                dataset = TensorDataset(kcs_cat.to(self.device), answer_cat.to(self.device), difficulty_cat.to(self.device))
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                pred_knowledge_level = torch.tensor([])
                for batch_kcs_cat, batch_answer_cat, batch_difficulty_cat in data_loader:
                    batch_kcs_cat = batch_kcs_cat.to(self.device)
                    batch_answer_cat = batch_answer_cat.to(self.device)
                    batch_difficulty_cat = batch_difficulty_cat.to(self.device)

                    pred_knowledge_piece = self.predict(batch_kcs_cat, batch_answer_cat, batch_difficulty_cat)
                    detach_pred_knowledge_piece =  pred_knowledge_piece[:,-1:].to('cpu').detach()
                    
                    pred_knowledge_level = torch.cat([pred_knowledge_level, detach_pred_knowledge_piece], dim=0)
                # pred_knowledge_level = self.predict(kcs_cat, answer_cat, difficulty_cat)[:,-1:]
                return pred_knowledge_level
            
            else:
                raise Exception('inputs shape or dimension are not valid. ')

    def get_sequential_knowledge_state(self, KCs, answer, difficulty, batch_size=64):
        if self.init_knowledge_state is None:
            self.get_knowledge_state()
        
        self.hist_knowledge_state['KCs'] = KCs
        self.hist_knowledge_state['answer'] = answer
        self.hist_knowledge_state['difficulty'] = difficulty
        self.hist_knowledge_state['knowledge_states'] = [self.init_knowledge_state.to('cpu').detach().view(-1)]

        if len(KCs) == len(answer) == len(difficulty) and (KCs.ndim * answer.ndim*difficulty.ndim)==1:
            len_KCs = len(KCs)
            for i in tqdm(range(len_KCs)):
                kcs_hist = KCs[:(i+1)]
                answer_hist = answer[:(i+1)]
                difficulty_hist = difficulty[:(i+1)]

                # print(kcs_hist, answer_hist, difficulty_hist)
                hist_knowledge_state = self.get_knowledge_state(kcs_hist, answer_hist, difficulty_hist, batch_size)
                self.hist_knowledge_state['knowledge_states'].append(hist_knowledge_state.to('cpu').detach().view(-1))
            return self.hist_knowledge_state['knowledge_states']
        else:
            raise Exception('inputs shape or dimension are not valid.')



# ktl = KnowledgeTracing_LSTM(knowledge_graph)

# num_tokens = 12000
# batch_size = 10
# max_seq_len = 1000
# kcs = torch.randint(0, num_tokens, (batch_size, max_seq_len))
# kcs = torch.randint(-1, num_tokens, (batch_size, max_seq_len))
# ans = torch.tensor(np.random.choice([-1,1,0], size=(batch_size, max_seq_len)), dtype=torch.float32)


# kcs = torch.tensor([[3,4,3],[10,24,5]])
# kcs = torch.tensor([3,4,3])

# kcs = torch.tensor(3)
# ans = torch.tensor(np.random.choice([-1,1,0], size=kcs.shape), dtype=torch.float32)
# diff = torch.rand_like(kcs.type(torch.float32))

# torch.tanh(ktl(kcs,ans, diff))


#######################
# (LSTM Tutorial)
# lstm = nn.LSTM(input_size=5, hidden_size=20, num_layers=3, batch_first=True)
# hidden_out, (last_out, cell_out) = lstm(torch.rand((10,100,5)))
# hidden_out.shape, last_out.shape, cell_out.shape


########################################################################################




# KnowledgeTracing
class KnowledgeTracing_LSTM_MidSchool(nn.Module):
    def __init__(self, knowledge_graph, KCs_size=12000,
                 KCs_embed_dim = 32, difficulty_embed_dim=8, ans_embed_dim = 8,

                 preKnowledgeWeight=0.7, postKnowledgeWeight=0.2,
                 preKnowledgeDepth=40, preKnowledgeCapa=150,
                 postKnowledgeDepth=40, postKnowledgeCapa=150,
                 rnn_layers=3, hidden_dim=64, output_dim=1, device='cpu'):
        super().__init__()
        self.preKnowledgeWeight = preKnowledgeWeight
        self.postKnowledgeWeight = postKnowledgeWeight
        self.output_dim = output_dim
        self.knowledge_graph = knowledge_graph

        self.kcs_midschool = kcs_midschool.copy()
        self.kcs_to_chapter_midschool = kcs_to_chapter_midschool.copy()
        self.ch_to_action = ch_to_action.copy()
        self.action_to_ch =  action_to_ch.copy()
        self.kc_to_idx_all =  kc_to_idx_all.copy()
        self.kc_to_idx =  kc_to_idx.copy()

        self.ids_KCs = torch.tensor([self.kcs_midschool]*3).type(torch.int64).view(-1,1)
        self.ids_answer = torch.zeros_like(self.ids_KCs).type(torch.float32)
        self.ids_difficulty = (torch.tensor([-1,0,1]).view(-1,1) * torch.ones_like(torch.tensor(self.kcs_midschool).view(-1)) ).type(torch.float32).view(-1,1)
        
        self.init_knowledge_state = None
        self.hist_knowledge_state = {}
        self.device = device

        # embedding
        self.KCs_graph_embedding = KCsEmbedding(KCs_size=KCs_size, embed_dim=KCs_embed_dim, knowledge_graph=knowledge_graph, 
                preKnowledgeWeight=preKnowledgeWeight, postKnowledgeWeight=postKnowledgeWeight,
                preKnowledgeDepth=preKnowledgeDepth, preKnowledgeCapa=preKnowledgeCapa, postKnowledgeDepth=postKnowledgeDepth, postKnowledgeCapa=postKnowledgeCapa)
        
        self.AnserEmbedding = AnserEmbedding(embed_dim=ans_embed_dim, hidden_dim=ans_embed_dim*2)

        self.difficulty_embedding = DifficultyEmbedding(input_dim=1, embed_dim=difficulty_embed_dim, hidden_dim=difficulty_embed_dim*2)

        self.rnn_embed_dim = self.KCs_graph_embedding.embed_dim + self.AnserEmbedding.embed_dim + self.difficulty_embedding.embed_dim

        self.rnn_layer = nn.LSTM(input_size= self.rnn_embed_dim,
                     hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True, bias=True,
                     dropout=0,)

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim, 16)
            ,nn.ReLU()
            ,nn.Linear(16, 4)
            ,nn.ReLU()
            ,nn.Linear(4, output_dim)
        )

    def forward(self, KCs, answer, difficulty):
        with torch.no_grad():
            self.device = KCs.device
        
        KCs_embed = self.KCs_graph_embedding(KCs)       # (batch, seq, KCs_embedding)
        ans_embed = self.difficulty_embedding(answer)    # (batch, seq, ans_embedding)
        difficulty_embed = self.difficulty_embedding(difficulty)    # (batch, seq, difficulty_embedding)

        embedding = torch.cat([KCs_embed, ans_embed, difficulty_embed], dim=-1)    # (batch, seq, KCs + difficulty embedding)

        # RNN Input Dimension에 맞추기 위함
        emb_ndim = embedding.ndim
        if emb_ndim < 3:
            for _ in range(3-emb_ndim):
                embedding = embedding.unsqueeze(0)

        # lstm
        hidden_out, (last_out, cell_out) = self.rnn_layer(embedding)
        # hidden_out.shape      # hidden_out: 각 타임스텝에서 LSTM이 생성한 출력 (batch, seq, output_dim)
        # last_out.shape        # last_out: 마지막 타임스텝에서의 hidden state 값 (num_layers, batch, output_dim)
        # cell_out.shape        # cell_out: 마지막 타임스텝에서의 cell state 값 (num_layers, batch, output_dim)

        # fc layer for prediction
        fc_out = self.fc_layer(hidden_out)

        # RNN Output을 Input Dimension에 맞추기 위함
        output = fc_out
        if emb_ndim < 3:
            for _ in range(3-emb_ndim):
                output = output.squeeze(0)
        
        if self.output_dim == 1:
            return output.squeeze(-1)
        else:
            return tuple([t.squeeze(-1) for t in torch.chunk(output, self.output_dim, dim=-1)])

    def predict(self, KCs, answer, difficulty):
        with torch.no_grad():
            output = self.forward(KCs, answer, difficulty)
            if type(output) == tuple:
                return tuple([(torch.tanh(t)/2+0.5) for t in output])
            else:
                return torch.tanh(output)/2 + 0.5
    
    def get_knowledge_state(self, KCs=None, answer=None, difficulty=None, batch_size=64):
        if KCs is None and answer is None and difficulty is None:
            self.init_knowledge_state = self.predict(self.ids_KCs.to(self.device), self.ids_answer.to(self.device), self.ids_difficulty.to(self.device))
            return self.init_knowledge_state
        else:
            if KCs.shape == answer.shape == difficulty.shape and (KCs.ndim * answer.ndim*difficulty.ndim)==1:
                kcs_broadcast = KCs.expand(self.ids_KCs.shape[0], KCs.shape[-1])
                answer_broadcast = answer.expand(self.ids_answer.shape[0], answer.shape[-1])
                difficulty_broadcast = difficulty.expand(self.ids_difficulty.shape[0], difficulty.shape[-1])

                kcs_cat = torch.cat([kcs_broadcast, self.ids_KCs], dim=-1)
                answer_cat = torch.cat([answer_broadcast, self.ids_answer], dim=-1)
                difficulty_cat = torch.cat([difficulty_broadcast, self.ids_difficulty], dim=-1)

                batch_size = batch_size
                dataset = TensorDataset(kcs_cat.to(self.device), answer_cat.to(self.device), difficulty_cat.to(self.device))
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                pred_knowledge_level = torch.tensor([])
                for batch_kcs_cat, batch_answer_cat, batch_difficulty_cat in data_loader:
                    batch_kcs_cat = batch_kcs_cat.to(self.device)
                    batch_answer_cat = batch_answer_cat.to(self.device)
                    batch_difficulty_cat = batch_difficulty_cat.to(self.device)

                    pred_knowledge_piece = self.predict(batch_kcs_cat, batch_answer_cat, batch_difficulty_cat)
                    detach_pred_knowledge_piece =  pred_knowledge_piece[:,-1:].to('cpu').detach()
                    
                    pred_knowledge_level = torch.cat([pred_knowledge_level, detach_pred_knowledge_piece], dim=0)
                # pred_knowledge_level = self.predict(kcs_cat, answer_cat, difficulty_cat)[:,-1:]
                return pred_knowledge_level

            else:
                raise Exception('inputs shape or dimension are not valid. ')

    def get_sequential_knowledge_state(self, KCs, answer, difficulty, batch_size=64):
        if self.init_knowledge_state is None:
            self.get_knowledge_state()
        
        self.hist_knowledge_state['KCs'] = KCs
        self.hist_knowledge_state['answer'] = answer
        self.hist_knowledge_state['difficulty'] = difficulty
        self.hist_knowledge_state['knowledge_states'] = [self.init_knowledge_state.to('cpu').detach().view(-1)]
        
        if len(KCs) == len(answer) == len(difficulty) and (KCs.ndim * answer.ndim*difficulty.ndim)==1:
            len_KCs = len(KCs)
            for i in tqdm(range(len_KCs)):
                kcs_hist = KCs[:(i+1)]
                answer_hist = answer[:(i+1)]
                difficulty_hist = difficulty[:(i+1)]

                # print(kcs_hist, answer_hist, difficulty_hist)
                hist_knowledge_state = self.get_knowledge_state(kcs_hist, answer_hist, difficulty_hist, batch_size)
                self.hist_knowledge_state['knowledge_states'].append(hist_knowledge_state.to('cpu').detach().view(-1))
            return self.hist_knowledge_state['knowledge_states']
        else:
            raise Exception('inputs shape or dimension are not valid.')


























class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000, dropout=0):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        
        # Positional Encoding을 저장할 텐서 생성
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        # div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        # 짝수 인덱스: sin, 홀수 인덱스: cos
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 차원에 맞추기 위해 (1, max_len, embed_dim) 형태로 차원 확장
        # pe = pe.unsqueeze(0)
        
        # 등록하여 학습 중 업데이트되지 않도록 설정
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 입력에 Positional Encoding 추가
        x = x + self.pe[:x.shape[-2]]
        return self.dropout(x)


# pe = PositionalEncoding(8)
# pe(torch.rand(6,8)).shape
# pe(torch.rand(10,5,8)).shape

# -------------------------------------------------------------------------------------------

# # KnowledgeTracing
# class KnowledgeTracing_Transformer(nn.Module):
#     def __init__(self, knowledge_graph, KCs_size=12000, KCs_embed_dim=32, difficulty_embed_dim=8,
#                  preKnowledgeWeight=1, postKnowledgeWeight=0.3,
#                  preKnowledgeDepth=40, preKnowledgeCapa=150,
#                  postKnowledgeDepth=40, postKnowledgeCapa=150,
#                  output_dim=1,
#                  num_heads=4, max_len=10000, n_blocks=3):
#         super().__init__()
#         self.output_dim = output_dim

#         self.KCs_graph_embedding = KCsEmbedding(KCs_size=KCs_size, embed_dim=KCs_embed_dim, knowledge_graph=knowledge_graph, 
#                 preKnowledgeWeight=preKnowledgeWeight, postKnowledgeWeight=postKnowledgeWeight,
#                 preKnowledgeDepth=preKnowledgeDepth, preKnowledgeCapa=preKnowledgeCapa, postKnowledgeDepth=postKnowledgeDepth, postKnowledgeCapa=postKnowledgeCapa)

#         self.difficulty_embedding = DifficultyEmbedding(input_dim=1, embed_dim=difficulty_embed_dim, hidden_dim=difficulty_embed_dim*2)

#         # positional encoding
#         self.transformer_embed_dim = self.KCs_graph_embedding.embed_dim + self.difficulty_embedding.embed_dim

#         self.positional_encoding = PositionalEncoding(self.transformer_embed_dim,
#                                                     max_len=max_len, dropout=0)

#         # transformer
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_embed_dim, nhead=num_heads, dim_feedforward=self.transformer_embed_dim*2, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=n_blocks)

#         self.fc_layer = nn.Sequential(
#             nn.Linear(self.transformer_embed_dim, 32)
#             ,nn.ReLU()
#             ,nn.Linear(32, 8)
#             ,nn.ReLU()
#             ,nn.Linear(8, output_dim)
#         )

#     def forward(self, KCs, difficulty):
#         KCs_embed = self.KCs_graph_embedding(KCs)       # (batch, seq, KCs_embedding)
#         difficulty_embed = self.difficulty_embedding(difficulty)    # (batch, seq, difficulty_embedding)

#         embedding = torch.cat([KCs_embed, difficulty_embed], dim=-1)    # (batch, seq, KCs + difficulty embedding)

#         # transformer Input Dimension에 맞추기 위함
#         emb_ndim = embedding.ndim
#         if emb_ndim < 2:
#             for _ in range(2-emb_ndim):
#                 embedding = embedding.unsqueeze(0)

#         # positional encoding
#         embedding_add_position = self.positional_encoding(embedding)

#         # masking : for no influence from future inputs 
#         seq_len = embedding_add_position.shape[-2]

#         attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(embedding_add_position.device)
#         attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))

#         # src_key_padding_mask for attention to apply padding for sequence length
#         src_key_padding_mask = torch.zeros(embedding_add_position.shape[:2], dtype=torch.bool, device=embedding_add_position.device)

#         # transformer
#         transformer_output = self.transformer_encoder(embedding_add_position, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

#         # fc layer for prediction
#         fc_out = self.fc_layer(transformer_output)

#         # transformer Output을 Input Dimension에 맞추기 위함
#         output = fc_out
#         if emb_ndim < 2:
#             for _ in range(2-emb_ndim):
#                 output = output.squeeze(0)
        
#         if self.output_dim == 1:
#             return output.squeeze(-1)
#         else:
#             return tuple([t.squeeze(-1) for t in torch.chunk(output, self.output_dim, dim=-1)])



# class KnowledgeTracing_Transformer(nn.Module):
#     def __init__(self, knowledge_graph, KCs_size=12000, KCs_embed_dim=32, difficulty_embed_dim=8,
#                  preKnowledgeWeight=1, postKnowledgeWeight=0.3,
#                  preKnowledgeDepth=40, preKnowledgeCapa=150,
#                  postKnowledgeDepth=40, postKnowledgeCapa=150,
#                  output_dim=1,
#                  num_heads=4, max_len=10000, n_blocks=3):
#         super().__init__()
#         self.output_dim = output_dim

#         self.KCs_graph_embedding = KCsEmbedding(KCs_size=KCs_size, embed_dim=KCs_embed_dim, knowledge_graph=knowledge_graph, 
#                 preKnowledgeWeight=preKnowledgeWeight, postKnowledgeWeight=postKnowledgeWeight,
#                 preKnowledgeDepth=preKnowledgeDepth, preKnowledgeCapa=preKnowledgeCapa, postKnowledgeDepth=postKnowledgeDepth, postKnowledgeCapa=postKnowledgeCapa)

#         self.difficulty_embedding = DifficultyEmbedding(input_dim=1, embed_dim=difficulty_embed_dim, hidden_dim=difficulty_embed_dim*2)

#         self.transformer_embed_dim = self.KCs_graph_embedding.embed_dim + self.difficulty_embedding.embed_dim
#         self.positional_encoding = PositionalEncoding(self.transformer_embed_dim, max_len=max_len, dropout=0)

#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_embed_dim, nhead=num_heads, dim_feedforward=self.transformer_embed_dim*2, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=n_blocks)

#         self.fc_layer = nn.Sequential(
#             nn.Linear(self.transformer_embed_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.ReLU(),
#             nn.Linear(8, output_dim)
#         )

#     def forward(self, KCs, difficulty):
#         batch_size, seq_len = KCs.size()
#         outputs = []

#         for t in range(seq_len):
#             # Slice up to current time step
#             KCs_slice = KCs[:, :t+1]
#             difficulty_slice = difficulty[:, :t+1]

#             # Embed the sliced inputs
#             KCs_embed = self.KCs_graph_embedding(KCs_slice)  # (batch, t+1, KCs_embedding)
#             difficulty_embed = self.difficulty_embedding(difficulty_slice)  # (batch, t+1, difficulty_embedding)
#             embedding = torch.cat([KCs_embed, difficulty_embed], dim=-1)  # (batch, t+1, transformer_embed_dim)

#             # Positional Encoding
#             embedding_add_position = self.positional_encoding(embedding)

#             # Create causal mask for the current sliced sequence
#             attn_mask = torch.triu(torch.ones(t+1, t+1), diagonal=1).to(embedding.device)
#             attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))

#             # Apply Transformer to current sequence slice
#             transformer_output = self.transformer_encoder(embedding_add_position, mask=attn_mask)

#             # Only take the last output (current time step prediction)
#             fc_out = self.fc_layer(transformer_output[:, -1, :])
#             outputs.append(fc_out)

#         # Concatenate predictions across all time steps
#         outputs = torch.cat(outputs, dim=1)

#         if self.output_dim == 1:
#             return outputs.squeeze(-1)
#         else:
#             return tuple([t.squeeze(-1) for t in torch.chunk(outputs, self.output_dim, dim=-1)])


# ktt = KnowledgeTracing_Transformer(knowledge_graph)

# num_tokens = 12000
# batch_size = 10
# max_seq_len = 1000
# kcs = torch.randint(0, num_tokens, (batch_size, max_seq_len))
# kcs = torch.tensor([[3,4,3],[10,24,5]])
# kcs = torch.tensor([3,4,3])
# kcs = torch.tensor(3)

# diff = torch.rand_like(kcs.type(torch.float32))
# ktt(kcs,diff)