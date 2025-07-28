import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Model Definition (same as used during training)
# -------------------------------
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels*i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=5):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels), UpsampleBlock(num_channels))
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        initial = self.initial(x)
        out = self.residuals(initial)
        out = self.conv(out) + initial
        out = self.upsamples(out)
        return self.final(out)

# -------------------------------
# Load the model
# -------------------------------
@st.cache_resource
def load_model(model_path="models\super_res_gen_600.pth", device='cpu'):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)

# -------------------------------
# Image processing
# -------------------------------
def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def postprocess(tensor):
    image = tensor.squeeze().cpu().detach().clamp(0, 1)
    image = transforms.ToPILImage()(image)
    return image

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🖼️ Super-Resolution with ESRGAN")
st.markdown("Upload a **low-resolution image** to upscale it using your trained ESRGAN model.")

uploaded_file = st.file_uploader("Choose a low-resolution image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Low-Resolution Input", use_container_width=True)

    with st.spinner("Upscaling..."):
        input_tensor = preprocess(input_image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_image = postprocess(output_tensor)

    st.image(output_image, caption="High-Resolution Output", use_container_width=True)
    st.success("✅ Super-resolution completed!")
    st.download_button("Download Output", output_image.tobytes(), file_name="output.png", mime="image/png")
