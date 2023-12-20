# Creating Your Own Dataset for 6D Pose Estimation

This guide outlines the steps and requirements for creating a custom dataset suitable for 6D pose estimation tasks.

## High level requirements
* Camera
* Interesting object
* Access to a printer
* Some decent compute

We will print a Charuco board to calibrate your camera and then use the same board to create a labeled dataset with the pose labels for your object.

## Label Files Structure

Our label files include 21 ground-truth values, derived as follows:
* 9 points corresponding to the centroid and corners of the 3D object model.
* Class label for each cell.
* Size encoding with 2 additional numbers representing the range in the x and y dimensions.

This results in a format of 9x2+1+2 = 21 numbers.

## Label Format

Each of the 21 numbers corresponds to specific data points:
0. Class Label
1. Centroid Coordinates: x0 (x-coordinate), y0 (y-coordinate)
3. Corner Coordinates: From x1, y1 (first corner) to x8, y8 (eighth corner)
19. Object Size Range: x range, y range
21. Focal length x
22. Focal length y
23. Sensor width (in optical this can just be the image size)
24. Sensor height (in optical this can just be the image size)
25. Focal offset x (u0)
26. Focal offset y (v0)
27. Image width
28. Image height
29. Object rotation vector (Rodriques) [3x1]
32. Object translation vector [3x1]

Note: Coordinates are normalized by image width and height (x / image_width, y / image_height).

## Tips for Training on Your Own Dataset

To train a model on your own dataset, you can mirror the LINEMOD dataset's structure. Ensure the following elements are included for each object:

1. Image Files Folder
2. Label Files Folder (Refer to [this link] for label creation guidance. Consider using the ObjectDatasetTools toolbox for ground-truth labels.)
3. Training Image Filenames (train.txt)
4. Test Image Filenames (test.txt)
5. 3D Object Model (.ply file, units in meters)
6. Segmentation Masks Folder (Optional, for background robustness)

## Configuration Adjustments
Ensure these configurations are tailored to your dataset:

* Diameter ("diam" value): Set to the diameter of your object model in the data configuration file.

