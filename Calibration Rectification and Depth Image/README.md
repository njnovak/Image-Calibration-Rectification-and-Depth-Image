This project focuses on stereovision. To run the output, you will need to have numpy, matplotlib, and opencv installed on your system.
Make sure the artroom, cehss, and ladder folders all contain im0.png and im1.png.

To run, simply run the [main.py](https://github.com/njnovak/Image-Calibration-Rectification-and-Depth-Image/blob/a3a2d6756ecd7b0d4f412b6f7d01bede2d93870f/Calibration%20Rectification%20and%20Depth%20Image/main.py) file. Via command line, this would look like `$python3 main.py`

It will prompt you for an input upon running. Select 1, 2, or 3 for the desired dataset.

It will save the resulting images into the corresponding files depending on the dataset selected.

Sometimes, main.py will fail. This is due to the RANSAC algorithm selecting bad points. Since RANSAC is random, this may be resolved by running the program a few more times.

Any more apparently issues, related or otherwise, should be directed to [nnovak@umd.edu], maintainer.
