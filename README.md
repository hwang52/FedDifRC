# FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning

This is an official implementation of the following paper:
> Huan Wang, Haoran Li, Huaming Chen, Jun Yan, Jiahua Shi, Jun Shen. *"FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning"*. IEEE/CVF International Conference on Computer Vision (ICCV), ICCV 2025.
---

**Abstract:** Federated learning aims at training models collaboratively across participants while protecting privacy. However, one major challenge for this paradigm is the data heterogeneity issue, where biased data preferences across multiple clients, harming the model's convergence and performance. In this paper, we first introduce powerful diffusion models into the federated learning paradigm and show that diffusion representations are effective steers during federated training. To explore the possibility of using diffusion representations in handling data heterogeneity, we propose a novel diffusion-inspired Federated paradigm with Diffusion Representation Collaboration, termed FedDifRC, leveraging meaningful guidance of diffusion models to mitigate data heterogeneity. The key idea is to construct text-driven diffusion contrasting and noise-driven diffusion regularization, aiming to provide abundant class-related semantic information and consistent convergence signals. On the one hand, we exploit the conditional feedback from the diffusion model for different text prompts to build a text-driven contrastive learning strategy. On the other hand, we introduce a noise-driven consistency regularization to align local instances with diffusion denoising representations, constraining the optimization region in the feature space. In addition, FedDifRC can be extended to a self-supervised scheme without relying on any labeled data. We also provide a theoretical analysis for FedDifRC to ensure convergence under non-convex objectives. The experiments on different scenarios validate the effectiveness of FedDifRC and the efficiency of crucial components.

---

Here is an example to run FedDifRC on CIFAR-10 with noniid_factor=0.05 & imb_factor=0.1:


```python
python3 main_FedDifRC.py --data_name cifar10 --num_classes 10 --non_iid_alpha 0.05 --imb_factor 0.1
```