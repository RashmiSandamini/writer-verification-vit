## ✍️ Quick Writer Verification with Automatic Feature Extraction

This repository contains my main contribution to the final year research project titled **Exam Candidate Verification Through Handwritten Artifacts**. 
I was responsible for designing and developing Module 2 – Quick Writer Verification with Automatic Feature Extraction, which performs text-independent handwriting verification using deep learning techniques and texture-based representations. 

### 🔧 Module Overview
This module verifies whether two handwriting samples originate from the same writer or not, even under variations such as writing speed.
There are 2 modes supported in this module.
- Standard mode verification - One sample per writer 
- Two speed mode verification - Uses both normal and fast handwriting to handle intra-writer variability.

### 🛠️ Key Features

#### Texture-Based Representation
Developed a pipeline to convert handwritten documents into compact texture images by minimizing irrelevant white spaces. This approach was inspired by prior works, *Writer Verification Using Texture-Based Features* by Hanusiak et al. and *Personal Identification Based on Handwriting* by Said et al.

The input to this pipeline is a line-removed, binarized handwriting sample. First, lines are segmented using horizontal projection, and then words are segmented using vertical projection. The detected words are placed onto a new canvas while reducing gaps between lines and words. This results in a compact visual representation of the handwriting. Finally, the texture image is split into patches. The pipeline is illustrated below.

<p align='center'>
  <img src="assets/texture creation pipeline.png" alt="Texture Creation Pipeline" width="600"/>
</p>

#### Siamese Network Architecture
The verification framework is built on a Siamese neural network, which is designed to learn and compare feature representations of two inputs. In our case, the Siamese network compares feature embeddings of two handwriting texture patches to determine whether they originate from the same writer. To extract these features, several pretrained models were evaluated, including VGG16, ResNet18, ResNet50, ViT224, and ViT384. These models were fine-tuned on our dataset, and ViT384 was selected for its superior performance.

A projection head consisting of three dense layers was added, with output dimensions of 512, 256, and 128 respectively. Each dense layer is followed by Layer Normalization and the GELU activation function. The output is then L2-normalized. The resulting embeddings are compared using Euclidean distance, a lower distance means the samples are likely from the same writer and a higher distance means they are likely from different writers.

<p align='center'>
  <img src="assets/overall siamese architecture.png" alt="Overall Siamese Network Architecture" width="600"/>
</p>

##### Standard Mode of Verification
In the standard verification mode, the system is designed to operate under the constraint of having only one handwriting sample per writer. To enhance the robustness of verification during inference, from each handwriting sample, four textures are generated. Each texture is independently passed through the feature extractor and projection head to obtain a feature embedding, as described in the architecture shown above. These embeddings are then averaged to form a single representative embedding for that document. The final verification decision is made by computing the Euclidean distance between the averaged embeddings of the two samples. This distance score is then thresholded to determine whether the samples originate from the same writer.

##### Two Speed Mode of Verification
In the two-speed mode, there should be two handwriting samples per writer, one written at a normal speed and the other at a fast speed, which results in four documents per verification instance. Similar to the standard verification mode, four textures are generated per document, feature embeddings are computed, and the embeddings are averaged to produce a single representation for each document. From these averaged embeddings, six pairwise comparisons are made:
- Two intra-writer comparisons - normal vs. fast samples from the same writer.
- Four inter-writercomparisons - cross comparisons between known and questioned samples across speeds.

Each pair is processed by the Siamese network, the same architecture that was used in standard mode of verification to produce a dissimilarity score. These six scores are then passed into a feedforward neural network which produces the final verification decision. This allows the model to exploit variability patterns rather than relying solely on static visual features. The architecture of this mode is presented below.

<p align='center'>
  <img src="assets/two speed verification architecture.png" alt="Two Speed Verification Architecture" width="600"/>
</p>

The feedforward neural network consists of of three fully connected layers with ReLU activations applied to the first two layers. The architecture is as follows,an input layer of size 6 (corresponding to the six pairwise distances), a hidden layer of size 4, a second hidden layer of size 2, and a final output layer of size 1. The final output is a single logit value representing the dissimilarity score between the two sets of handwriting samples. A sigmoid activation is applied during inference to obtain a probability score. This neural network was trained using the same writers that was used to finetune the feature extractors.
