from net import *
from PIL import Image
from torchvision import transforms
from tic_tac_toe import *


transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

layer1 = 2*np.pi*(1-transform(Image.open("data/layer1.png")).squeeze().to(device))
layer2 = 2*np.pi*(1-transform(Image.open("data/layer2.png")).squeeze().to(device))
layer3 = 2*np.pi*(1-transform(Image.open("data/layer3.png")).squeeze().to(device))
model = DON(layer1, layer2, layer3).to(device)

env = TicTacToe(1)
state = env.reset()

intensity = np.zeros(9)
position = []

for step in range(1000):
    img = Image.fromarray(state)
    input = transform(transforms.Resize((Ny, Nx))(img)).squeeze().to(device)
    output = model(input)

    intensity[0] = np.sum(output.cpu().numpy()[130:240, 130:240])
    intensity[1] = np.sum(output.cpu().numpy()[130:240, 490:600])
    intensity[2] = np.sum(output.cpu().numpy()[130:240, 850:960])
    intensity[3] = np.sum(output.cpu().numpy()[490:600, 130:240])
    intensity[4] = np.sum(output.cpu().numpy()[490:600, 490:600])
    intensity[5] = np.sum(output.cpu().numpy()[490:600, 850:960])
    intensity[6] = np.sum(output.cpu().numpy()[850:960, 120:240])
    intensity[7] = np.sum(output.cpu().numpy()[850:960, 490:600])
    intensity[8] = np.sum(output.cpu().numpy()[850:960, 850:960])

    for i in range(9):
        if i in position:
            intensity[i] = 0

    action = np.argmax(intensity)
    position.append(action)

    state, reward, done, info = env.step(action)
    env.render()

    if done == True:
        position = []
        state = env.reset()
    

env.close()