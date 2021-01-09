import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets.numpy()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, h_n = self.gru(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN().to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(-1, 28, 28)
        output = rnn(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x.to(device)).cpu()
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.cpu().data.numpy(), '| accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].to(device).view(-1, 28, 28)).cpu()
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y)
print(test_y[:10])
img = test_x[:10].view(-1, 1, 28, 28)
save_image(img, "img/1.jpg", nrow=5)
