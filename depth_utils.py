import numpy as np
import matplotlib.pyplot as plt

depth = np.load("C:/Users/Lenovo/Desktop/Diplmoma_all/dataset/depth/depth/28_depth.npy")

plt.imshow(depth, cmap="viridis")
plt.colorbar(label="Depth value")
plt.title("Depth Map Visualization")
plt.axis("off")
plt.show()
