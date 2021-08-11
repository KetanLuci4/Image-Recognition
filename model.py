import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import maincode

training_data = np.load("training_data_4.npy", allow_pickle=True)
np.random.shuffle(training_data)

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# print(len(train_X), len(test_X))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)        #-> Tsize 46
        self.pool1 = nn.MaxPool2d(2,2)         #-> Tsize 23
        self.conv2 = nn.Conv2d(10,20,5)        #-> Tsize 19
        self.pool2 = nn.MaxPool2d(2,2)          #-> Tsize 8
        self.conv3 = nn.Conv2d(20,32,5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*8*8,2000)
        self.fc2 = nn.Linear(2000,1500)
        self.fc3 = nn.Linear(1500,1101)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.Softmax(self.fc3(x))
        return x

net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


BATCH_SIZE = 5
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in range(0, len(train_X), BATCH_SIZE):
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")

correct = 0
total = 0
with torch.no_grad():
    for i in range(len(test_X)):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))
