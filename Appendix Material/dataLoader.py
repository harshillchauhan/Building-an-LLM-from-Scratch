import torch
from torch.utils.data import Dataset, DataLoader

#^ DataLoader
#? A DataLoader in PyTorch is a utility that makes it easy to feed data into your model during training or evaluation.

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

#! The difference between train data and test data is that train data is used to teach the model patterns, while test data is used to evaluate how well the model performs in unseen data.

class ToyDataset (Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    #? Instructions for retrieving exactly one data record and the corresponding label
    def __getitem__(self, index):       
        one_x = self.features[index]    
        one_y = self.labels[index]      
        return one_x, one_y 
    
    #? Instructions for returning the total length of the dataset
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))
print(len(test_ds))

#? This is added to get the exact same result from the reference material. If we remove this, then the output generated would be random.
torch.manual_seed(123)

#? Creating a DtaLoader object for train data
train_loader = DataLoader(
    #? Using the ToyDataset instance containing test features and labels
    dataset = train_ds, 
    #? Instructing to load 2 samples per batch
    batch_size = 2,
    #? Instructing to keep the data order random
    shuffle = True,
    #? Loading data using the main process (No parallel workers)
    num_workers = 0
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0
)

for idx, (x,y) in enumerate(train_loader) :
    print(f"Batch {idx+1}:", x,y)

#? OUTPUT : Batch 1: tensor([[ 2.3000, -1.1000],
#?                  [-0.9000,  2.9000]]) tensor([1, 0])
#?          Batch 2: tensor([[-1.2000,  3.1000],
#?                  [-0.5000,  2.6000]]) tensor([0, 0])
#?          Batch 3: tensor([[ 2.7000, -1.5000]]) tensor([1])

#! As we see based on the output, the train_loader iterates over the training dataset, visiting each training example exactly once. This is knows as training epoch.

#? EPOCH : An epoch means one complete pass through the entire training dataset.

#! We have set num_workers = 0, which means data loading will be done in the main process and not in seperate worker process.

#? This at large scale is problamatic as it can cause significant slowdowns during model training.
#? In contrast when num_workers is set to a number greater than 0, multiple worker processes are launched to load data in parallel, freeing the main process to focus on training our model and better utilizing our system resources.
