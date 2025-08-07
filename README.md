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
Developed a pipeline to convert handwritten documents into compact texture images by minimizing irrelevant white spaces. This was inspired by prior works, "Writer verification using texture-based features" by Hanusiak et al. and "Personal identification based on handwriting"
by Said et al. Input for this pipeline was a line removed, binarized handwriting sample. Then the lines are segmented using horizontal projection, and words are segmented using vertical projection. After that those detected 
words are placed in a new cavas while reducing the gaps between the words and the lines. By doing so, we were able to obtain a compact version of handwriting from the sample. After that the big texture is splited into patches. The pipeline can be seen in below image.

<p align='center'>
  <img src="assets/" alt="Texture Creation Pipeline" width="600"/>
</p>
