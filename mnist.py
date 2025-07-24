import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pdb
import matplotlib.pyplot as plt
# Download and load MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Access one image and label
image, label = mnist_dataset[0]   # image shape: torch.Size([1, 28, 28])

# Convert to NumPy array and flatten
vector = image.numpy().flatten()  # shape: (784,)

# Convert to numpy and reshape if needed
image_np = image.squeeze().numpy()  # shape: (28, 28)

# Plot the image
plt.imshow(vector.reshape(1, 784)[:, 128:256], cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

pdb.set_trace()