import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import ast
import cv2



path = "data/raw/signverod_dataset/images/nist_r0304_01.png"
img = Image.open(path).convert("RGB")  # Convert to RGB
draw = ImageDraw.Draw(img)
draw.rectangle([500, 500, 900, 900], outline="green", width=50)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()
