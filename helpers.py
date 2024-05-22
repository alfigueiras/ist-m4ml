import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

#Graphics
import plotly.graph_objects as go

from u_net import SimplifiedUNet
from imp_u_net import ImprovedSimplifiedUNet


T = 500
IMG_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def linear_noise_schedule(start=0.0001, end=0.02, steps=T):
    return torch.linspace(start=start, end=end, steps=steps)


betas = linear_noise_schedule().to(DEVICE)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def reverse_transform_tensor(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.clamp(-1, 1)),
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(image)


def sample(model, n):
    """
    Implementation of the algorithm 2 of the paper (Sampling algorithm)
    """
    # print(f"Sampling {n} new images")

    # plt.figure(figsize=(20, 3))
    # plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    t_list = list(range(1, T, stepsize))

    model.eval()
    with torch.no_grad():
        # Creates random noise image
        x = torch.randn((n, 3, IMG_SIZE, IMG_SIZE)).to(DEVICE)
        for i in list(range(1, T))[::-1]:
            t = (torch.ones(n) * i).long().to(DEVICE)

            # Predicts noise at timestep t for all images random noise images in x
            predicted_noise = model(x, t)

            alpha = alphas[t][:, None, None, None]
            alpha_cumprod = alphas_cumprod[t][:, None, None, None]
            beta = betas[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Removing noise from the image
            x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(
                beta) * noise

            # Backward process figure
            if i in t_list:
                if len(x.shape) == 4:
                    img = x[0, :, :, :].cpu()
                # plt.subplot(1, num_images + 1, int(i / stepsize) + 1)
                # new_img = reverse_transform_tensor(img)
                # plt.imshow(new_img)

    # plt.show()
    model.train()

    # Transform to deafult color intensity values
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


def sample_imp(model, n, guidance_level=0.7, class_samples=None, n_classes=10):
    """
    Implementation of the algorithm 2 of the paper (Sampling algorithm)
    """
    # print(f"Sampling {n} new images")

    # plt.figure(figsize=(20, 3))
    # plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    t_list = list(range(1, T, stepsize))

    model.eval()
    with torch.no_grad():
        # Creates random noise image
        x = torch.randn((n, 3, IMG_SIZE, IMG_SIZE)).to(DEVICE)
        if class_samples is None:
            if n_classes == 1:
                # Cars belong to class 1, generate labels corresponding to cars observations
                class_samples = torch.ones(n).int().to(DEVICE)
            elif n_classes == 5:
                # Generate labels belonging to the set of previously chosen classes
                possible_classes = torch.tensor([0, 1, 3, 7, 8])
                indexes = torch.randint(0, n_classes, (n,))
                class_samples = possible_classes[indexes].int().to(DEVICE)
            # print(class_samples)
        for i in list(range(1, T))[::-1]:
            t = (torch.ones(n) * i).long().to(DEVICE)

            # Predicts noise at timestep t for all noisy images in x
            predicted_noise = model(x, t, class_samples)
            predicted_noise_wout_cfg = model(x, t, None)

            # Interpolates between predicted noise obtained without information from classes and with classes
            final_predicted_noise = torch.lerp(predicted_noise_wout_cfg, predicted_noise, guidance_level)

            alpha = alphas[t][:, None, None, None]
            alpha_cumprod = alphas_cumprod[t][:, None, None, None]
            beta = betas[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Removing noise from the image
            x = (1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * final_predicted_noise) + torch.sqrt(
                beta) * noise).float()

            # x=x.clamp(-1,1)

            # Backward process figure
            if i in t_list:
                if len(x.shape) == 4:
                    img = x[0, :, :, :].cpu()
                # plt.subplot(1, num_images + 1, int(i / stepsize) + 1)
                # new_img = reverse_transform_tensor(img)
                # plt.imshow(new_img)

    # plt.show()
    model.train()
    # Transform to deafult color intensity values
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def get_images(dl, n):
    images = []
    for batch in dl:
        batch_images = batch[0]
        images.extend(batch_images)
        if len(images) >= n:
            break
    images = torch.stack(images)
    return images


def sample_model(model_number, epoch, num_images):
    model = SimplifiedUNet()
    model.load_state_dict(torch.load(os.path.join(f"DDPM{model_number}", "models", f"DDPM{model_number}", f"checkpoint{epoch}.pt")))
    model.to(DEVICE)
    return sample(model, num_images)


def get_fid_values(model_number, epochs, real_images):
    metric = FrechetInceptionDistance(feature=64, normalize=True)
    values = []
    i = 0
    n = len(epochs)
    for epoch in epochs:
        print(f'\r{i}/{len(epochs)}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}", f"{epoch}.pt"))
        metric.update(real_images, real=True)
        metric.update(gen_images.cpu(), real=False)
        values.append(metric.compute())
        metric.reset()
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])


def get_inception_values(model_number, epochs):
    metric = InceptionScore(normalize=True)
    values = []
    i = 0
    n = len(epochs)
    for epoch in epochs:
        print(f'\r{i}/{n}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}", f"{epoch}.pt"))
        values.append(metric(gen_images.cpu())[0])
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def sample_imp_model(model_number, epoch, num_images, n_classes, guidance_level=0.7):
    model = ImprovedSimplifiedUNet(n_classes=10)
    model.load_state_dict(torch.load(os.path.join(f"DDPM{model_number}_IMP", "models", f"DDPM{model_number}_IMP", f"checkpoint{epoch}.pt")))
    model.to(DEVICE)
    return sample_imp(model, num_images, guidance_level, n_classes=n_classes)


def get_imp_fid_values(model_number, epochs, real_images):
    metric = FrechetInceptionDistance(feature=64, normalize=True)
    values = []
    i = 0
    n = len(epochs)
    for epoch in epochs:
        print(f'\r{i}/{n}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}_IMP", f"{epoch}.pt"))
        metric.update(real_images, real=True)
        metric.update(gen_images.cpu(), real=False)
        values.append(metric.compute())
        metric.reset()
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])


def get_imp_inception_values(model_number, epochs):
    metric = InceptionScore(normalize=True)
    values = []
    i = 0
    n = len(epochs)
    for epoch in epochs:
        print(f'\r{i}/{n}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}_IMP", f"{epoch}.pt"))
        values.append(metric(gen_images.cpu())[0])
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])


def get_gl_imp_fid_values(model_number, real_images):
    guidance_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    metric = FrechetInceptionDistance(feature=64, normalize=True)
    values = []
    i = 0
    n = len(guidance_levels)
    for gl in guidance_levels:
        print(f'\r{i}/{n}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}_IMP", f"gl{gl}.0.pt"))
        metric.update(real_images, real=True)
        metric.update(gen_images.cpu(), real=False)
        values.append(metric.compute())
        metric.reset()
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])


def get_gl_imp_inception_values(model_number):
    guidance_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    metric = InceptionScore(normalize=True)
    values = []
    i = 0
    n = len(guidance_levels)
    for gl in guidance_levels:
        print(f'\r{i}/{n}', end='')
        gen_images = torch.load(os.path.join("images", f"DDPM{model_number}_IMP", f"gl{gl}.0.pt"))
        values.append(metric(gen_images.cpu())[0])
        i += 1
    print(f'\r{n}/{n}', end='')
    return np.array([float(v) for v in values])

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def plot_line_chart(file_name, x, y, title="", x_title="", y_title="", trace1=""):
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{trace1}'))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    fig.update_layout(
        title=title,
        template='plotly',
        plot_bgcolor='white',
        width=800,
        height=400
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    # fig.write_image(f'plots/{file_name}.png')
    return fig


def plot_2_line_chart(file_name, x, y1, y2, title="", x_title="", y_title="", trace1="", trace2=""):
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{trace1}', marker=dict(color='red')))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'{trace2}', marker=dict(color='deepskyblue')))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    fig.update_layout(
        title=title,
        template='plotly',
        plot_bgcolor='white',
        width=800,
        height=400
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    # fig.write_image(f"plots/{file_name}.png", engine="kaleido")
    return fig


def plot_4_line_chart(x, y1, y2, y3, y4, title="", x_title="", y_title="", trace1="", trace2="", trace3="", trace4=""):
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{trace1}', marker=dict(color='firebrick')))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'{trace2}', marker=dict(color='red')))

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name=f'{trace3}', marker=dict(color='cornflowerblue')))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y4, mode='lines', name=f'{trace4}', marker=dict(color='deepskyblue')))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    fig.update_layout(
        title=title,
        template='plotly',
        plot_bgcolor='white',
        width=800,
        height=400
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig
