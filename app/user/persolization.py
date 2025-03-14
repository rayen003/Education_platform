from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd



class Dataset(DataSet):
    def __init__(self, data):
        super().__init__()
        self.data = pd.read_csv(data)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        