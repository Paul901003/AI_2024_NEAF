import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


#Load data
train_x = []
train_x_std = []
train_y = []
folder_name = ['Yes', 'No']
i = 0
for folder in folder_name:
    path = 'Data/'+ str(folder) +'/'
    for root, dirs, files in walk(path):
        for f in files:
            filename = path + f
            print(filename)
            
            acc = scipy.io.loadmat(filename)
            acc = acc['tsDS'][:, 1].tolist()[0:7500]
            train_x.append(acc)
            train_x_std.append(np.std(acc))
            
            if folder == 'Yes':    
                train_y.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = 'fig/original/Yes_'
            
            if folder == 'No':
                train_y.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = 'fig/original/No_'
                
            # # plt.clf()
            # plt.figure(figsize=(7,4))
            # plt.plot(acc, 'b-', lw=1)
            # plt.title(title + str(i+1))
            # plt.xlabel('Samples')
            # plt.ylabel('Acceleration')
            # plt.savefig(saved_file_name + str(i+1) + '.png')                
            # # plt.show()
            # i = i + 1

train_x = np.array(train_x_std)
train_y = np.array(train_y)

print(train_x)
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x.reshape(-1, 1))

# Convert to PyTorch tensors
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.long)

class Net(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output):
        super(Net, self).__init__() # To inherit things from Net, the standard process must be added
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden_2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden_3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden_4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.hidden_5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.out = torch.nn.Linear(n_hidden5, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = self.out(x)
        return x

net = Net(n_feature=1, n_hidden1=100, n_hidden2=100, n_hidden3=100, n_hidden4=100, n_hidden5=100, n_output=2) # define the network
print(net) # network architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss() # 已經包含了 sigmoid or softmax

for t in range(50):
    out = net(train_x_tensor)
    loss = loss_func(out, train_y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        prediction = torch.max(out, 1)[1]
    y_pred = prediction.numpy()
    y_true = train_y_tensor.numpy()

print('Prediction: \t', y_pred)
print('Ground Truth \t', y_true)

cf_m = confusion_matrix(y_true, y_pred)
print('Confusion Matrix: \n', cf_m)

tn, fp, fn, tp = cf_m.ravel()
accuracy = (tn + tp) / (tn + fp + fn + tp)
print('Accuracy: ', accuracy)

# Optional: Visualization
plt.scatter(train_x, train_y, c=y_pred, s=100, lw=0, cmap='RdYlGn')
plt.xlabel('Standard Deviation (Normalized)')
plt.ylabel('Chatter (0 = No, 1 = Yes)')
plt.title(f'Accuracy = {accuracy:.2f}')
plt.show()