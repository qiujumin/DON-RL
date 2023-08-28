from net import *
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import CUSTOM_MOVEMENT
from PIL import Image
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from torchvision import transforms

layer1 = nn.Parameter(torch.normal(0, 1, size=(Ny, Nx)))
layer2 = nn.Parameter(torch.normal(0, 1, size=(Ny, Nx)))
layer3 = nn.Parameter(torch.normal(0, 1, size=(Ny, Nx)))

model = DON(layer1, layer2, layer3).to(device)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, CUSTOM_MOVEMENT)
policy = PPO.load("data/ppo_supermariobros", env=env)
vec_env = policy.get_env()
obs = vec_env.reset()


def target_intensity(action):
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()])
    if action == 0:
        return transform(Image.open('data/run.png')).squeeze()
    elif action == 1:
        return transform(Image.open('data/jump.png')).squeeze()
    else:
        return transform(Image.open('data/down.png')).squeeze()


for step in range(1000):
    action, _states = policy.predict(obs, deterministic=True)
    img = transforms.Grayscale()(torch.from_numpy(obs)/255)
    x = transforms.Resize((Ny, Nx))(img).squeeze().to(device)
    y = target_intensity(action[0]).to(device)
    y_pred = model(x)

    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

    if step % 100 == 99:
        print("epoch {:>3d}: loss = {:>8.3f}".format(step, loss))

        for _, parameters in enumerate(model.parameters()):
            phase = parameters.cpu().detach().numpy()
            phase = np.where(phase > np.pi, phase - 2 * np.pi, phase)
            phase = np.where(phase < -np.pi, phase + 2 * np.pi, phase)

            plt.imsave(f"data/layer{_+1}.png", phase, cmap="gray_r")
