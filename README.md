<div align="center">  

# YOLOv5-6D Pose: Advancing 6-DoF Instrument Pose Estimation in Variable X-Ray Imaging Geometries
6-DoF Pose estimation based on the [YOLO framework](https://github.com/ultralytics/yolov5) and tailored to our application of instrument pose estimation in X-Ray images.  
</div>

<h3 align="center">
  <a href="">Project Page</a> |
  <a href="https://ieeexplore.ieee.org/document/10478293">Paper</a> |
  <a href="">arXiv</a> |
</h3>

<p float="left">
<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/iron.gif" width=45% height=50%>
<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/cat.gif" width=45% height=50%>

<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/screw_train.gif" width=90%>
<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/cube.gif" width=45% height=50%>
<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/phantom.gif" width=45% height=50%>
</p>


<!--[teaser](results/images/architecture.png)-->

<div align="center">  

## Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov5-6d-advancing-6-dof-instrument-pose/6d-pose-estimation-on-linemod)](https://paperswithcode.com/sota/6d-pose-estimation-on-linemod?p=yolov5-6d-advancing-6-dof-instrument-pose)
<p >
<img src="https://github.com/cviviers/YOLOv5-6D-Pose/blob/main/results/images/acc_vs_speed.png" width=60% height=60%> 
</p>
</div>

## Quick Start
<details>
<summary>Setup</summary>

Clone repo and install [requirements.txt](https://github.com/cviviers/YOLOv5-6D-Pose/blob/master/requirements.txt) in a
[**Python>=3.9.0**](https://www.python.org/) environment, including
[**PyTorch==1.9**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/cviviers/YOLOv5-6D-Pose.git  # clone
cd YOLOv5-6D-Pose
pip install -r requirements.txt  # install

cd utils
python3 ./setup.py build_ext --inplace # To compute ADD-S metric

```
Alternatively, build the [dockerfile](https://github.com/cviviers/YOLOv5-6D-Pose/blob/master/docker) included or pull our image [sudochris/yolo6dpose:v2](https://hub.docker.com/repository/docker/sudochris/yolo6dpose/general).

Download the [weights](https://drive.google.com/drive/folders/11BW41xO3R1UBnc2Dx1xA3CPbYPGTrfHQ?usp=drive_link) and [data](https://drive.google.com/drive/folders/13pkdF4KpqXWXbEIkiAcHoO-YUr_CWN-L?usp=drive_link)

</details>
<details>
<summary>Inference</summary>

```bash
python detect.py --weights path/to/weights/linemod/cat/best.pt --img 640 --conf 0.25 --source ../data/LINEMOD/cat/JPEGImages --static-camera configs/linemod/linemod_camera.json --mesh-data path/to/data/LINEMOD/cat/cat.ply
```

</details>

<details>
<summary>Training</summary>

```bash
python train.py --batch 32 --epochs 5000 --cfg yolov5xv6_pose_bifpn.yaml --hyp configs/hyp.single.yaml --weights yolov5x.pt --data configs/linemod/benchvise.yaml --rect --cache --optimizer Adam 
```

</details>

## Detailed Setup

For a detailed description on how to set up YOLOv5-6D, follow our tutorial [tutorial](https://github.com/cviviers/YOLOv5-6D-Pose/blob/master/tutorial.ipynb).
We have also created an easy [guide](https://github.com/cviviers/YOLOv5-6D-Pose/blob/master/data_curation/CreatingDataset.md) on how to create a single object dataset for using with YOLOv5-6D.


## TODO List <a name="todos"></a>
- [x] Release code: code will be released upon paper acceptance.
- [x] Release weights: will be released upon paper acceptance.
- [ ] Linemod occlusion results
- [ ] Multi-object Pose
- [ ] Release code & results for YOLOv8-6D pose

## Citation <a name="citation"></a>

Please consider citing our [paper](https://ieeexplore.ieee.org/document/10478293) if the project helps your research with the following BibTex:

```
@ARTICLE{10478293,
  author={Viviers, Christiaan G.A. and Filatova, Lena and Termeer, Maurice and De With, Peter H.N. and Sommen, Fons van der},
  journal={IEEE Transactions on Image Processing}, 
  title={Advancing 6-DoF Instrument Pose Estimation in Variable X-Ray Imaging Geometries}, 
  year={2024},
  keywords={X-ray instrument detection;6-DoF pose estimation;surgical vision;imaging geometry;deep learning},
  doi={10.1109/TIP.2024.3378469}}
```

## License <a name="license"></a>

All assets and code are under the [AGPL-3.0 License](./LICENSE) - in line with YOLOv5 - unless specified otherwise.
