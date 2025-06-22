# digit_generator_app.py
# Full MNIST digit generator using PyTorch + Streamlit

import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Model Definition ---
class DigitGenerator(nn.Module):
    def __init__(self):
        super(DigitGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10 + 100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # 0–1 range for grayscale pixels
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x).view(-1, 28, 28)

# --- Load Trained Model ---
def load_model():
    model = DigitGenerator()
    model.load_state_dict(torch.load("digit_gen_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Image Generation Function ---
def generate_images(model, digit, num_images=5):
    images = []
    for _ in range(num_images):
        noise = torch.randn(1, 100)
        label = torch.zeros(1, 10)
        label[0, digit] = 1
        with torch.no_grad():
            img = model(noise, label).squeeze(0).numpy()
            images.append(img)
    return images

# --- Streamlit App ---
st.title("MNIST Digit Generator")
st.write("Select a digit (0–9) to generate 5 handwritten-style images.")

selected_digit = st.selectbox("Digit", list(range(10)))

if st.button("Generate Images"):
    model = load_model()
    imgs = generate_images(model, selected_digit)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    st.pyplot(fig)

# Save this file and run with: `streamlit run digit_generator_app.py`
