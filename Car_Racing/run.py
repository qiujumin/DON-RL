from net import *
import gym
from PIL import Image, ImageFilter
from torchvision import transforms


def gaussian_noise(img, mean, var):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    output = img + noise

    if output.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    output = np.clip(output, low_clip, 1.0)
    output = np.uint8(output * 255)

    return output


transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

layer1 = 2*np.pi*(1-transform(Image.open("data/layer1.png")).squeeze().to(device))
layer2 = 2*np.pi*(1-transform(Image.open("data/layer2.png")).squeeze().to(device))
layer3 = 2*np.pi*(1-transform(Image.open("data/layer3.png")).squeeze().to(device))
model = DON(layer1, layer2, layer3).to(device)

env = gym.make("CarRacing-v0")
env.reset()

for step in range(30):
    env.step([0, 0.4, 0])
    env.render()

for step in range(1200):
    # img = Image.fromarray(env.state).filter(ImageFilter.GaussianBlur(radius=10))
    # img = Image.fromarray(gaussian_noise(env.state, mean=-0.02, var=0.02))
    img = Image.fromarray(env.state)
    input = transform(transforms.Resize((Ny, Nx))(img)).squeeze().to(device)
    output = model(input)

    left = np.sum(output.cpu().numpy()[460:570, 330:430])
    right = np.sum(output.cpu().numpy()[460:570, 1180:1300])

    if 0.01*(right-left) > 10:
        steering = 0.001*(right-left)
    elif 0.01*(right-left) < -10:
        steering = 0.001*(right-left)
    else:
        steering = 0

    env.step([steering, 0.01, 0])
    env.render()

env.close()
