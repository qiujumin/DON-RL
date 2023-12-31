from net import *
from PIL import Image
from torchvision import transforms
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import CUSTOM_MOVEMENT


transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

layer = [torch.from_numpy(np.loadtxt(f"data/layer{i}.csv", delimiter=",")).to(device) for i in range(3)]
model = DON(layer).to(device)

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, CUSTOM_MOVEMENT)
state = env.reset()

for step in range(1000):
    img = Image.fromarray(state)
    input = transform(transforms.Resize((Ny, Nx))(img)).squeeze().to(device)
    output = model(input)

    jump = np.sum(output.cpu().numpy()[180:290, 490:590])
    down = np.sum(output.cpu().numpy()[790:890, 490:590])
    run = np.sum(output.cpu().numpy()[460:570, 870:980])

    if jump>down and jump>run:
        for step in range(20):
            state, reward, done, info = env.step(1)
            env.render() 
        for step in range(1):
            state, reward, done, info = env.step(0)
    elif down>jump and down>run:
        for step in range(20):
            state, reward, done, info = env.step(2)
            env.render() 
        for step in range(1):
            state, reward, done, info = env.step(0)
    else:
        state, reward, done, info = env.step(0)

    env.render() 

env.close()
