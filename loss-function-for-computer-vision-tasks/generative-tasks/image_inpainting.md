# **Image Inpainting**

Image inpainting is a computer vision and image processing technique used to restore or fill in missing or damaged parts of an image. The primary objective is to reconstruct these missing areas in a way that results in a visually realistic and semantically consistent completed image. Deep learning techniques, including **CNN-based** and **ViT-based** models, are widely used for both image and video inpainting.

To achieve high-quality results, multiple **loss functions** are typically combined during training. Each loss plays a role in addressing different aspects of the inpainting objective. Common loss functions include:


* **Mean Absolute Error (L1 Loss):** Ensures pixel-level accuracy.
* **Adversarial Loss:** Promotes realism by encouraging outputs that fool a discriminator.
* **Perceptual Loss:** Preserves high-level semantic and perceptual details.
* **Reconstruction Loss:** Ensures the restored content aligns with the original structure.
* **Style Loss:** Focuses on capturing texture, color, and artistic style.
* **Feature Map Loss:** Preserves consistency across intermediate feature representations.



## **Examples of Methods**

1. **Partial Convolution**

   * Uses **masked convolutions** to prioritize valid pixel regions during training.
   * Employs **L1 Loss** for pixel accuracy, **Style Loss** for texture alignment, and **Perceptual Loss** for semantic integrity.

2. **Contextual Transformer Network (CTN)**

   * A ViT-based framework that models relationships between corrupted and intact regions.
   * Uses **Reconstruction, Perceptual, Adversarial, and Style Losses**.

3. **Mask-aware Transformer (MAT)**

   * Designed for large-hole inpainting in high-resolution images.
   * Utilizes **Perceptual** and **Adversarial Losses**.

4. **T-former**

   * A ViT-based approach with efficient long-range modeling.
   * Combines **Reconstruction, Perceptual, Adversarial, and Style Losses**.

5. **TransCNN-HAE**

   * Hybrid CNN + ViT architecture for handling varied image damage.
   * Uses **Reconstruction, Perceptual, and Style Losses**.

6. **CoordFill**

   * Employs **Fourier Convolution (FFC)**-based generation for capturing wide receptive fields.
   * Uses **Perceptual, Adversarial, and Feature Matching Losses** to enhance high-frequency texture synthesis.

7. **Continuous Mask-aware Transformer (CMT)**

   * Learns to inpaint using **continuous masks**.
   * Optimized with **Reconstruction, Perceptual, and Adversarial Losses**.



## **Loss Function Categories in Image Inpainting**

* **Contextual-based Losses**
  Preserve semantic information to ensure inpainted regions blend naturally with their surroundings.
  *Examples: L1 Loss, Reconstruction Loss.*

* **Style-based Losses**
  Capture global semantics and artistic textures rather than pixel-level precision.
  *Examples: Perceptual Loss, Style Loss, Adversarial Loss.*

* **Structural-based Losses**
  Emphasize contextual harmony and structural coherence, ensuring that inpainted regions maintain alignment with surrounding structures.


By combining losses across these categories, deep learning methods can achieve **realistic, context-aware, and visually coherent inpainting results**.

---


# **Mean Absolute Error (MAE) Loss**

### **Definition**

The **Mean Absolute Error (MAE)**, or **L1 Loss**, measures the **average absolute difference** between predicted values and actual ground-truth values. Unlike squared error losses, it treats all errors linearly, without disproportionately penalizing larger deviations.



### **Mathematical Formula**

For predictions $`\hat{y}_i`$ and true labels $y_i$ over $N$ samples:

$`L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} | y_i - \hat{y}_i |`$


### **Intuition**

* MAE calculates **how far predictions are from targets on average**.
* It is **robust to outliers** compared to Mean Squared Error (MSE), since it does not square the errors.
* Encourages **sparse error minimization**, which can be useful in tasks where exact precision is less important than overall trend alignment.



### **Practical Use Cases**

* **Regression tasks** where robustness to outliers is required.
* **Image inpainting** for maintaining pixel-level precision (often combined with perceptual or adversarial losses).
* **Time series forecasting** to evaluate average deviation without overweighting extreme anomalies.
* **Medical imaging** where small deviations are tolerable, but robustness is crucial.


MAE is simple, interpretable, and widely used — though in practice it is often combined with other losses (like **MSE**, **Perceptual Loss**, or **Adversarial Loss**) to balance robustness with sensitivity.

---


# **Adversarial Loss**

### **Definition**

Adversarial Loss arises from the **minimax game** between two models in GANs:

* A **Generator (G)** that tries to produce realistic samples.
* A **Discriminator (D)** that tries to distinguish between real and generated samples.

The loss quantifies how well the generator fools the discriminator and how well the discriminator separates real from fake.



### **Mathematical Formula**

The original GAN loss:

$`\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]`$

Where:

* $`D(x)`$ = probability that input $x$ is real.
* $`G(z)`$ = generated sample from latent noise $z$.
* $`p_{data}`$ = true data distribution.
* $`p_z`$ = prior distribution (e.g., Gaussian) for noise.

For training:

* **Discriminator Loss**:

$`L_D = -\Big( \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))] \Big)`$

* **Generator Loss**:

$`L_G = -\mathbb{E}[\log D(G(z))]`$



### **Intuition**

* The **discriminator** improves by learning to detect fakes.
* The **generator** improves by making fakes look more real.
* Over time, the generator produces outputs indistinguishable from real data, achieving equilibrium where $`D(x) \approx 0.5`$ for both real and fake samples.



### **Practical Use Cases**

* **Image generation** (GANs, StyleGAN, BigGAN).
* **Image-to-image translation** (Pix2Pix, CycleGAN).
* **Text-to-image generation** (AttnGAN, DALL·E, Stable Diffusion with adversarial refinements).
* **Super-resolution** (SRGAN).
* **Inpainting** (GauGAN, Contextual GANs).


Adversarial Loss is powerful because it **learns implicit data distributions** without explicitly modeling them, making it central to generative deep learning.


---


# **Perceptual Loss**

### **Definition**

Perceptual Loss measures differences between images not at the **pixel level**, but in a **high-level feature space** extracted by a pre-trained network (e.g., VGG, ResNet). Instead of comparing raw pixel values, it compares semantic features, making it more aligned with human perception of image quality.



### **Mathematical Formula**

Given a pre-trained network $`\phi`$ and feature maps from layer $`l`$:

$`L_{perc} = \frac{1}{N_l} \sum_{i=1}^{N_l} \| \phi_l(y_i) - \phi_l(\hat{y}_i) \|_2^2`$

Where:

* $`y_i`$ = ground truth image.
* $`\hat{y}_i`$ = generated image.
* $`\phi_l(\cdot)`$ = activation map from layer $l$.
* $`N_l`$ = number of elements in that feature map.



### **Intuition**

* Pixel-wise losses (L1, L2) may produce **blurry results**, especially in generative tasks.
* Perceptual Loss ensures the generated image **“looks” realistic** by capturing **textures, edges, and semantics**.
* It allows models to generate outputs that are visually closer to human perception, even if pixel-level errors are higher.



### **Practical Use Cases**

* **Super-resolution** (SRGAN, ESRGAN).
* **Style transfer** (Neural Style Transfer uses style + perceptual losses).
* **Image-to-image translation** (Pix2Pix, CycleGAN).
* **Inpainting** (GauGAN, CoordFill).
* **Text-to-image & diffusion models** (used to refine realism).



Perceptual Loss is a **feature-based loss**, bridging the gap between **numerical similarity** and **visual quality**. It is often **combined with pixel-wise and adversarial losses** for balanced performance.

---


# **Reconstruction Loss**

### **Definition**

Reconstruction Loss measures how well a model can **reproduce its input** from an encoded or transformed representation. It is the core training signal in **autoencoders, variational autoencoders (VAEs), and many generative models**, ensuring that the learned latent space retains sufficient information to rebuild the original data.



### **Mathematical Formula**

For input $x$ and reconstructed output $`\hat{x}`$:

* Using **Mean Squared Error (MSE)**:

$`L_{rec} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2`$

* Using **Mean Absolute Error (MAE / L1 Loss)**:

$`L_{rec} = \frac{1}{N} \sum_{i=1}^N |x_i - \hat{x}_i|`$

Sometimes reconstruction loss is computed in **feature space** (similar to perceptual loss) instead of pixel space.



### **Intuition**

* Encourages the model to **retain and reconstruct essential details** of the input.
* If the reconstruction loss is low, the latent representation captures enough information.
* Balances **fidelity (pixel accuracy)** and **semantic integrity** depending on the choice of error function (L1, L2, perceptual, etc.).



### **Practical Use Cases**

* **Autoencoders (AE)** → enforce faithful input reconstruction.
* **Variational Autoencoders (VAE)** → combined with KL Divergence to regularize latent space.
* **Image inpainting** → ensures filled regions match the original input structure.
* **Super-resolution & denoising** → ensures reconstructed high-res/cleaned images preserve source details.
* **Speech/audio-to-image & cross-modal tasks** → ensures reconstructed modality aligns with input modality.


Reconstruction Loss is a **foundational loss function** — simple but essential — and is usually combined with **adversarial** and **perceptual losses** in modern generative frameworks.


---


# **Style Loss**

### **Definition**

Style Loss measures how well a generated image captures the **texture, patterns, and artistic style** of a reference style image. Instead of focusing on pixel-level similarity, it compares **feature correlations** between images using a **Gram matrix** derived from activations of a pre-trained network (e.g., VGG).



### **Mathematical Formula**

Given feature maps $`\phi_l(x)`$ from layer $l$ of a network:

1. **Gram Matrix** for layer $l$:

$`G_l(x) = \phi_l(x) \cdot \phi_l(x)^T`$

2. **Style Loss**:

$`L_{style} = \sum_{l} \frac{1}{4 N_l^2 M_l^2} \| G_l(\hat{y}) - G_l(y) \|_F^2`$

Where:

* $`G_l(x)`$ = Gram matrix of activations at layer $l$.
* $`N_l`$ = number of feature maps (channels).
* $`M_l`$ = number of elements per feature map (height × width).
* $y$ = style image, $`\hat{y}`$ = generated image.



### **Intuition**

* The **Gram matrix** captures **feature correlations** → representing **textures, colors, and patterns**, not object structure.
* Style Loss ensures that the generated image has **similar artistic properties** to the style image, even if the content differs.
* Complements **Content Loss** (which preserves structural information).



### **Practical Use Cases**

* **Neural Style Transfer** → blending content of one image with artistic style of another.
* **Image-to-image generation** → ensuring consistent artistic or texture properties across domains.
* **Super-resolution & inpainting** → preserving textures and fine-grained styles.
* **GAN-based image synthesis** → improving realism by aligning textures with real samples.



Style Loss is **style-aware but not content-aware**, making it ideal for tasks where texture and artistic similarity matter more than exact pixel accuracy.

---


# **Feature Map Loss**

### **Definition**

Feature Map Loss measures the difference between **feature representations** of a generated image and a target image at intermediate layers of a pre-trained neural network (e.g., VGG, ResNet). Unlike pixel-wise losses, it ensures that generated images capture **high-level structural and semantic information**.



### **Mathematical Formula**

Given feature maps $`\phi_l(x)`$ from layer $l$:

$`L_{fm} = \frac{1}{N_l} \sum_{i=1}^{N_l} \| \phi_l(y) - \phi_l(\hat{y}) \|_2^2`$

Where:

* $y$ = ground truth image.
* $`\hat{y}`$ = generated image.
* $`\phi_l(\cdot)`$ = feature map extracted from layer $l$.
* $`N_l`$ = number of elements in the feature map.



### **Intuition**

* Encourages generated images to match **semantic content and structure**, not just pixel values.
* Works as a **bridge between perceptual and style loss**:

  * **Perceptual loss** focuses on similarity in deep features.
  * **Style loss** focuses on feature correlations (Gram matrices).
  * **Feature map loss** directly aligns feature activations.
* Helps models produce sharper, semantically meaningful reconstructions.



### **Practical Use Cases**

* **Image inpainting** → ensures missing regions align semantically with the rest of the image.
* **Super-resolution** → encourages high-frequency structural fidelity.
* **GAN training (feature matching loss)** → stabilizes training by comparing intermediate discriminator features of real and fake samples.
* **Style transfer** → can be combined with style loss to enforce both structure and texture similarity.



Feature Map Loss is especially useful when combined with **adversarial** and **pixel-wise losses**, as it balances realism with semantic accuracy.

---

