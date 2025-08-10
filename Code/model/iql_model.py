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


import torch
import numpy as np
from six.moves import cPickle


train_path=f"{base_path}/code/iql"
dataset_path=f"{base_path}/dataset"

# Test class to load the model and perform inference
class IQLTest:
    def __init__(self, model_path):
        # Load the saved TorchScript model
        self.model = torch.jit.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def infer(self, state):
        # Convert state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            action = self.model(state_tensor)
        return action.cpu().numpy()

# Path to the saved model
# model_path = "/home/kimds929/MathProject/code/iql/saved_model/IQLtest/iql_11-12_model.pth"  # Replace with actual path
  # Replace with the actual path
# test_data = cPickle.load( open(f"{train_path}/test_11-12_set.pkl", 'rb') )
# test_data['action']=test_data['action'].apply(lambda x: x[0] + (x[1]+1)*102)
# Create an instance of the test class

# iql_test = IQLTest(model_path)

