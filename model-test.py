import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class LogisticRegressionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.linear(x)

model = LogisticRegressionMNIST()
model.load_state_dict(torch.load("mnist_logreg.pth"))
model.eval()

def predict_image(path):
    transform_custom = T.Compose([
        T.Resize((28,28)),   
        T.Grayscale(),       
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(path)
    tensor_img = transform_custom(img).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(tensor_img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

print("Predicted digit:", predict_image("my_digit.png"))