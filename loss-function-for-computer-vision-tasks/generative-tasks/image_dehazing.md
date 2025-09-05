### **Image Dehazing**

Image and video dehazing is an important area in computer vision that aims to **improve visual quality** by removing or reducing haze effects. Haze can arise from multiple sources: **atmospheric haze** (light scattering in the air, reducing contrast and saturation), **foggy haze** (dense atmospheric mist leading to diffusion), and **pollution haze** (smog or industrial emissions producing yellowish or brownish tones with reduced clarity). Environmental conditions such as rain, heat, and varying lighting can also contribute to haziness. Each type of haze influences both the **appearance and mood** of an image, requiring different dehazing strategies for effective restoration.


Over the years, numerous approaches have been proposed using CNNs, ViTs, and GANs, each adopting specific **loss functions** to guide high-quality restoration. For instance, 

- **Araji et al.** introduced a multi-scale representation network trained with **L1 loss** to maintain detail quality. 
- **HSMD-Net** further enhanced feature representation through hierarchical interaction modules, employing a **combination of Smooth L1, Perceptual, Adversarial, and Multiscale SSIM losses** to improve color and texture recovery. Similarly, 
- **FFA-Net** used attention mechanisms with **L1 loss** to boost reconstruction, while the transformer-based 
- **DehazeFormer** extended this idea with ViTs and L1 loss for better global representation. 
- Hybrid CNN–ViT approaches have also been proposed, leveraging **SSIM and MSE loss functions** for more robust training under varying hazy conditions.


Beyond CNN and ViT, GAN-based dehazing methods have also emerged. For example, **CycleGAN-based frameworks** employ **cycle-consistency loss, SSIM, and pixel-wise mean loss** to restore clean images from hazy inputs. These GAN-based strategies emphasize not only structural fidelity but also perceptual quality. Overall, the choice of **loss function is central to performance**—with pixel-wise losses (L1, MSE) ensuring accuracy, perceptual and SSIM-based losses enhancing realism, and adversarial/cycle-consistency losses driving naturalness in generated outputs.

---

