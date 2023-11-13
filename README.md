# Multi-Task Learning Framework

Welcome to the Multi-Task Learning Framework repository! This framework is designed for multi-task learning and incorporates three datasets: CelebA, NYUv2, and Cityscapes. Additionally, it includes six popular multi-task learning (MTL) loss weighting strategies, along with two novel contributions: Distributed Communicating Uncertainty Weighting and Monte Carlo Dropout.

## Datasets

### 1. CelebA
- **Original Paper:** [Deep Learning Face Attributes in the Wild](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **GitHub Repository:** [CelebA GitHub](https://github.com/switchablenorms/CelebAMask-HQ)

### 2. NYUv2
- **Original Paper:** [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)
- **GitHub Repository:** [NYU Depth V2 GitHub](https://github.com/ankurhanda/nyuv2-meta-data)

### 3. Cityscapes
- **Original Paper:** [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/)
- **GitHub Repository:** [Cityscapes GitHub](https://github.com/mcordts/cityscapesScripts)

## Multi-Task Learning Strategies

### Existing Strategies

1. **Uncertainty Weighting (UW)**
   - **Paper:** [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)
   - **GitHub Repository:** [Uncertainty Weighting GitHub](https://github.com/YannDubs/Deep_Multi_Task_Weighting)

2. **Random Loss Weighting (RLW)**
   - **Paper:** Not available
   - **GitHub Repository:** [Random Loss Weighting GitHub](https://github.com/yaohuaxu1994/RLW)

3. **Equal Weighting (EW)**
   - **Paper:** Not available
   - **GitHub Repository:** Not available

4. **CAGrad**
   - **Paper:** [CAGradientDescent: A New Optimization Algorithm for Deep Neural Networks](https://arxiv.org/abs/2004.02147)
   - **GitHub Repository:** [CAGradientDescent GitHub](https://github.com/lafeigong/CAGradientDescent)

5. **PCGrad**
   - **Paper:** [PCGrad: Inverse Curriculum Learning for Personalized Gradient-Based Training of Deep Models](https://arxiv.org/abs/2001.07466)
   - **GitHub Repository:** [PCGrad GitHub](https://github.com/chaoyuaw/pytorch-cifar-models)

6. **Geometric Loss Strategy (GLS)**
   - **Paper:** [Multi-task learning using uncertainty to weigh loss for scene geometry and semantics](https://openreview.net/forum?id=rJgX8JBYPS)
   - **GitHub Repository:** [Geometric Loss Strategy GitHub](https://github.com/ubikuity/GML_Multi_Task)

### Novel Contributions

7. **Distributed Communicating Uncertainty Weighting**

8. **Monte Carlo Dropout**

## Usage

This repository allows for seamless setup of a new dataset and integration of loss weighting strategies for multi-task learning. Follow these steps:

1. Clone the repository: `git clone https://github.com/Simon1307/MTL.git`
2. Navigate to the project directory: `cd MTL`
3. Set up your new dataset
4. Choose or implement loss weighting strategies for multi-task learning.
5. Customize and extend the framework to suit your specific needs.

Feel free to explore, contribute, and utilize this framework for your multi-task learning projects.

---
*Note: Please make sure to cite the original papers and repositories of the datasets and loss weighting strategies when using them in your work.*
