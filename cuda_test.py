import torch
import pickle as pkl
import os

# check if GPU is available, if True chooses to use it
cuda = torch.cuda.is_available()  
device = 'cuda' if cuda else 'cpu'
if cuda:
    print("Nvidia CUDA is available")
    torch.backends.cudnn.benchmark = True
print("Only CPU available")

# Test saving files
dir_results = 'results/'

dict_test = {'A': 1, 'B': [1, 3, 5], 'C': "Hey!"}
file_path = os.path.join(dir_results, 'test_save.pkl')

with open(file_path, 'wb') as f:
    pkl.dump(dict_test, f)

# check if results are saved correctly
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'rb') as f:
        dummy = pkl.load(f)
    print("Data was saved successfully.")
    # remove the test file
    os.remove(file_path)
else:
    print(f"Error: File '{file_path}' does not exist or is empty. The save was insuccesful")
