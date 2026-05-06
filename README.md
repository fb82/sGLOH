<h1 align="center">🎯🎭 sGLOH & sGLOH2 descriptors ⚙️🪛</h1>

<p align="center">
  <h2 align="center"><p>
    🔥 🐍 Python implementation </a> 🐍 🔥
  </p></h2>

<p align="center">
    <img src="https://github.com/fb82/sGLOH/blob/main/data/ET_sGOr2a.jpg" alt="example" height=200>
    <img src="https://github.com/fb82/sGLOH/blob/main/data/DC_sGOr2a.jpg" alt="example" height=200>
    <br>
    <em>HarrisZ+ and DoG + <b>sGLOH2</b> + Blob matching + DTM + RANSAC</em>
    <br>
    <ins>Fully handcrafted matching pipeline!</ins>
  
</p>

## What is it?
+ The shifting Gradient Orientation Local Histogram (sGLOH) descriptor is a SIFT-like descriptor where descriptor vectors on rotated patches are achieved by permutations of their elements, allowing several robust matching strategies according to the task.
+ The [original code](https://sites.google.com/view/fbellavia/research) was released in C, this is the Python implementation.

## Setup (including all the stuff to launch the demo)
Run from the terminal
```
git clone https://github.com/fb82/sGLOH.git
cd sGLOH
git submodule update --init --recursive
pip install -r requirements.txt
```
The current requirement file has been tested on Ubuntu 24.04. 

## Launch the demo
To use the demo with the default example image pair run from the terminal 
```
python ./demo.py
```
or with your image pair as
```
python ./demo.py <path of 1st image> <path of the 2nd image>
```
For further details or customizations please inspect the comments in ``demo.py``.

## Notes
+ Code is not optimized, especially the matching step can be slow. Increasing the ``max_n`` when computing the all pair distance table in ``sgloh_dist`` speeds up the code, but be careful to OOM issues.
+ The demo uses DoG and [HarrisZ+](https://github.com/fb82/HarrisZ) keypoints and [Blob matching + DTM](https://github.com/fb82/DTM) to compute the matches. In case of OOM issues you can select only one kind of keypoints. By default Blob Matching runs on CPU to avoid OOM, but if you have enough memory you can try on GPU. The demo allows alternatively the usage of the standard MNN.

## Where can I find more details?
+ [Rethinking the sGLOH descriptor](http://cvg.dsi.unifi.it/pdfs/sGLOH2_TPAMI.pdf) (TPAMI 2018)
+ [Keypoint descriptor matching with context-based orientation estimation](https://www.researchgate.net/publication/262770174_Keypoint_descriptor_matching_with_context-based_orientation_estimation) (Image and Vision Computing 2014)
