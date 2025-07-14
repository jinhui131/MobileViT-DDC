---


---

<h1 id="mobilevit-ddcenhanced-mobilevit-with-dilated-and-deformable-attention-for-intangible-cultural-heritage-embroidery-recognition">MobileViT-DDC:Enhanced MobileViT with Dilated and Deformable Attention for Intangible Cultural Heritage Embroidery Recognition</h1>
<p>This repo contains the official <strong>PyTorch</strong> code for MobileViT-DDC</p>
<h1 id="introduction">Introduction</h1>
<p><img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure1.jpeg" alt="enter image description here"><br>
Dilateformer Black<br>
<img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure2.jpeg" alt="enter image description here"><br>
defdilateformer Black<br>
<img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure4.jpeg" alt="enter image description here"></p>
<h3 id="key-features">Key Features:</h3>
<ul>
<li>In this paper, the Dilateformer Black replaces the ViT modules in the third and fourth stages, and the CBS module is introduced. This solves the problem of effectively capturing local structures and long-range dependencies in the shallow feature modeling stage of the model, while reducing redundant computations in the traditional self-attention mechanism. Experimental results show that this module not only reduces the model parameters but also significantly improves the accuracy.</li>
<li>To further enhance the adaptability of the network to complex deformations and fine-grained changes in the middle and high-level feature modeling stage, this paper proposes the DefDilate Block module. Based on the Dilate Block, this module incorporates the Deformable Convolution mechanism, dynamically adjusts the attention sampling positions, adaptively optimizes the local receptive field, and effectively enhances the model’s capability to model spatially deformed features. It is particularly suitable for dealing with the complex local structural changes present in ethnic minority embroidery patterns.</li>
<li>By redesigning the ViT module in MobileViT and integrating the advantages of Dilateformer and DefDilateformer, this paper proposes the **MobileViT - DDC ** model. Comprehensive comparative experiments prove that this model outperforms all benchmark comparison models.</li>
</ul>
<h3 id="method">Method</h3>
<p>Dilateformer<br>
<img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure3.jpeg" alt="enter image description here"><br>
Each head performs attention computation within a local window, centered at the red query point, using a distinct dilation rate to control the spacing of sampled keys and values. These varying dilation rates effectively enable different receptive field sizes (e.g., 3×3, 5×5, and 7×7), allowing the model to capture multi-scale contextual information. The outputs of all attention heads are then concatenated and passed through a linear projection to aggregate the information.</p>
<p>Defdilateformer<br>
<img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure5.jpeg" alt="enter link description here"><br>
DefDilateformer adds deformable convolution to Dilateformer, making attention point selection more flexible and enabling more effective recognition of complex embroidery images. A large number of ablation experiments demonstrate the superior performance of this module in the task of embroidery image recognition.</p>
<p>CBS<br>
<img src="https://github.com/jinhui131/MobileViT-DDC/blob/master/figures/figure6.jpeg" alt="enter image description here"><br>
CBS module obtains a global context representation via median pooling and broadcasts it to all tokens, enabling density compensation and enhanced feature robustness. Compared to average pooling, median pooling offers greater stability when handling outliers and noise, effectively improving the global perception capability of attention mechanisms under sparse structures.</p>
<h2 id="dependencies">Dependencies</h2>
<ul>
<li>Python 3.12</li>
<li>PyTorch == 2.5.1</li>
<li>torchvision == 0.20.1</li>
<li>numpy</li>
<li>timm == 0.5.4</li>
<li>yacs</li>
<li>typing</li>
</ul>
<h2 id="dataset">Dataset</h2>
<p>Basic Pattern Dataset (BPD) Women Clothing<br>
original：<br>
<a href="https://www.kaggle.com/rizzsum/datasets">https://www.kaggle.com/rizzsum/datasets</a><br>
Used in the paper：<a href="https://ieeexplore.ieee.org/document/9080687">https://ieeexplore.ieee.org/document/9080687</a><br>
Guizhou Embroidery:<br>
<a href="https://www.kaggle.com/datasets/jinhui1234/guizhou-embroidery">https://www.kaggle.com/datasets/jinhui1234/guizhou-embroidery</a></p>

