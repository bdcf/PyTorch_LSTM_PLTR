import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = pd.read_csv('PLTR.csv')
# We have to strip the whitespace and reverse the data as it currently goes 2025-2022.
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)
# As we are only forecasting the stocks price at close each day we will only use that data.
data = data[['Date', 'Close']]
# Convert the raw Date data to a Pandas datetime object.
data['Date'] = pd.to_datetime(data['Date'])

def prepLSTMDataFrame(df, n_steps):
    # Load each row as a dataframe for the LSTM.
    df = dc(df)
    df.set_index('Date', inplace=True)
    # Creates n_steps new columns with previous Close values, shifted back 1 to n_steps.
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    # If any columns are N/A then we get rid of them.
    df.dropna(inplace=True)
    return df
lookBack = 5
shiftDf = prepLSTMDataFrame(data, lookBack)
shiftDfNp = shiftDf.to_numpy()
# Scales the data between -1 and 1 as preprocessing to normalize data.
scaler = MinMaxScaler(feature_range=(-1, 1))
shiftDfNp = scaler.fit_transform(shiftDfNp)
features = shiftDfNp[:, 1:]
targets = shiftDfNp[:, 0]
# Flip the features so 1 is first and lookback is last.
features = dc(np.flip(features, axis=-1))
# Uses 95% to train and 5% to test
split = int(len(features) * 0.95)
fTrain, fTest, tTrain, tTest = features[:split], features[split:], targets[:split],targets[split:]
fTrain = torch.tensor(fTrain, dtype=torch.float32).unsqueeze(2)  
fTest = torch.tensor(fTest, dtype=torch.float32).unsqueeze(2)    
tTrain = torch.tensor(tTrain, dtype=torch.float32).unsqueeze(1)            
tTest = torch.tensor(tTest, dtype=torch.float32).unsqueeze(1)
# print(f'{fTest.shape}{fTrain.shape}{tTest.shape}{tTrain.shape}') to make sure the unsqueeze added an extra dimension [Testing]. We have to build a dataset of our train and test data of tuples for AI training and testing
class buildDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
trainDataset = buildDataset(fTrain, tTrain)
testDataset = buildDataset(fTest, tTest)
# We now will put our dataset into a data loader to normalize batch sizes for our normalized data.(friends don't let friends use batch sizes of >32)
batchSize = 16
# Set shuffle to true on training data so the AI does not get used to the training data order
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True) 
testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
# Now to make the LSTM training model.
class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numStackedLayers):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numStackedLayers = numStackedLayers
        # Define LSTM layer.
        self.lstm = nn.LSTM(inputSize, hiddenSize, numStackedLayers, batch_first=True)
        # Define fully connected Linear layer to map from hidden state to output.
        self.fc = nn.Linear(hiddenSize, 1)
        
    def forward(self, x):
        batchSize = x.size(0)
        # Initialie hidden and cell states with all zeroes.
        h0, c0 = torch.zeros(self.numStackedLayers, batchSize, self.hiddenSize).to(device), torch.zeros(self.numStackedLayers, batchSize, self.hiddenSize).to(device)
        # Pass the input through the LSTM layer.
        out, _ = self.lstm(x, (h0, c0))
        # Take the last time steps output for each sequence in the batch.
        out = self.fc(out[:, -1, :])
        return out
# Define the model with one feature, 4 neurons per LSTM layer, with one LSTM layer.
modelOne = LSTM(1, 4, 1)
modelOne.to(device)
learnRate = 0.001
epoch = 100
lossFunc = nn.MSELoss()
optFunc = torch.optim.Adam(modelOne.parameters(), lr= learnRate)

def trainEpoch():
    modelOne.train(True)
    print(f'Epoch: {epoch + 1}')
    runningLoss = 0.0
    # Loop through to get data from trainLoader batches
    for i, batch in enumerate(trainLoader):
        fbatch, tbatch = batch[0].to(device), batch[1].to(device)
        # Compute predictions.
        output = modelOne(fbatch)
        # Compute loss between predictions and the ground truth.
        loss = lossFunc(output, tbatch)
        runningLoss += loss
        # Zero out gradients from the previous step.
        optFunc.zero_grad()
        # Use a backpropogation pass to compute gradient.
        loss.backward()
        # Take a step in direction of gradient.
        optFunc.step()
        if i % 100 == 99:
            avgBatchLoss = runningLoss / 100
            print(f'Batch {i + 1}, Loss: {avgBatchLoss}')
            runningLoss = 0.0
    print()

def validateEpoch():
    modelOne.train(False)
    runningLoss = 0.0
    # Go through to get batches from testLoader
    for i, batch in enumerate (testLoader):
        fbatch, tbatch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = modelOne(fbatch)
            # Compute loss between predictions and the ground truth.
            loss = lossFunc(output, tbatch)
            runningLoss += loss
    avgBatchLoss = runningLoss / len(testLoader)
    print(f'Val Loss: {avgBatchLoss}')
for epoch in range(1, epoch):
    trainEpoch()
    validateEpoch()
with torch.no_grad():
    predict = modelOne(fTrain.to(device)).to('cpu').numpy()
tPredict = predict.squeeze()
dummyTensor = np.zeros((tPredict.shape[0], lookBack+1))
# Insert flattened predictions into first column of dummy array
dummyTensor[:, 0] = tPredict
# Get a deepcopy of first column
tPredict = scaler.inverse_transform(dummyTensor)[:, 0]

nTTrain = tTrain.squeeze().cpu().numpy()
dummyTensor = np.zeros((fTrain.shape[0], lookBack+1))
dummyTensor[:, 0] = nTTrain
nTTrain = scaler.inverse_transform(dummyTensor)[:, 0]

# Plot and present all the data predicted and actual from the LSTM model.
plt.plot(nTTrain, label='Actual Close')
plt.plot(tPredict, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
