# FashionVAE
Latent Space Exploration with Fashion MNIST. A Variational Autoencoder (VAE) to compress and visualize Fashion MNIST data in a 2D latent space using TensorFlow/Keras.

# FashionVAE 👗📉

**FashionVAE** is a deep learning project that implements a **Variational Autoencoder (VAE)** to learn a compressed 2D latent representation of the Fashion MNIST dataset. It enables the generation of new fashion items and visualizes the learned latent space structure.

---

## 🧠 What is a VAE?

A Variational Autoencoder is a type of generative model that learns the probability distribution of data in a lower-dimensional latent space. VAEs are useful for generating similar-looking images and for exploring how different latent vectors affect the generated output.

---

## 📦 Dataset

- **Fashion MNIST**: 70,000 grayscale images (28x28) across 10 categories (e.g., t-shirts, sneakers, bags).
- Comes preloaded with `keras.datasets`.

---

## 🚀 Features

- Encoder compresses images into a 2D latent space.
- Decoder reconstructs images from latent space vectors.
- Uses the **reparameterization trick** to allow backpropagation through stochastic layers.
- Includes a visualization function to **plot the entire latent space** and observe how image features vary.

---

## 📈 Model Architecture

- **Encoder**: Conv2D layers → Dense → μ and log(σ²)
- **Sampling Layer**: z = μ + σ * ε
- **Decoder**: Dense → Conv2DTranspose to reconstruct images
- **Loss**: Binary crossentropy + KL divergence

---

## 🧰 Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---
## 🔧 Setup


1. **Install dependencies**:

```
pip install tensorflow matplotlib numpy
```

2. **Run the script**:

```bash
python fashion_vae.py
```

---

## 🖼️ Sample Output

The script includes a function to visualize the 2D latent space:

```
plot_latent_space(vae, n=10, figsize=10)
```

Output:

> A 10x10 grid of generated fashion images where each image corresponds to a point in the latent space.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

* TensorFlow/Keras team for the APIs
* Yann LeCun & team for the Fashion MNIST dataset
* Kingma & Welling (2013) for VAE formulation


