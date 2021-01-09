import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

EPOCH = 1
BATCH_SIZE = 64
INPUT_SIZE = 28 * 28
LR = 0.01
DOWNLOAD_MNIST = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()

train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.
test_x = test_x.view(-1, 1, 28, 28)
test_y = test_data.targets.numpy()[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),                              # 16 * 28 * 28
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                # 16 * 14 * 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 32 * 14 * 14
            nn.LeakyReLU(),
            nn.MaxPool2d(2),                # 32 * 7 * 7
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.out(x)
        return x


cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        output = cnn(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x.to(device)).cpu()
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.cpu().data.numpy(), '| accuracy: %.2f' % accuracy)
            writer.add_scalar('loss', loss.cpu().data.numpy(), step)
            writer.add_scalar('accuracy', accuracy, step)

test_output = cnn(test_x[:10].to(device)).cpu()
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y)
print(test_y[:10])
