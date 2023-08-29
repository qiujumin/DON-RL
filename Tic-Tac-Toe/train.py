from net import *
from tic_tac_toe import *
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

env = TicTacToe(1)
policy = PPO.load("data/ppo_tictactoe", env=env)
vec_env = policy.get_env()
obs = vec_env.reset()

output = torch.zeros(Ny, Nx)


def target_intensity(action):
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()])
    point = transform(Image.open('data/point.png')).squeeze()
    size = point.size()
    output[action//3*size[0]:action//3*size[0]+size[0], action %
           3*size[1]:action % 3*size[1]+size[1]] = point

    return output


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
    # vec_env.render()

    if dones == True:
        output = torch.zeros(Ny, Nx)

    if step % 100 == 99:
        print("epoch {:>3d}: loss = {:>8.3f}".format(step, loss))

        for _, parameters in enumerate(model.parameters()):
            phase = parameters.cpu().detach().numpy()
            phase = np.where(phase > np.pi, phase - 2 * np.pi, phase)
            phase = np.where(phase < -np.pi, phase + 2 * np.pi, phase)

            plt.imsave(f"data/layer{_+1}.png", phase, cmap="gray_r")
