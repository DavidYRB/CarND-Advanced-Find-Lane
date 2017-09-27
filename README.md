There are two set of code. The first one is consist of different `.py` files whose names are `camera_calibration.py`, `image_gen.py`, `tracker.py`, `video_process.py`. Another is a `.ipynb` file which is easy to show the result of each section. It contains code of `camera_calibration.py`, `tracker.py`,  revised `image_gen.py` for single image process. The result of each stage of test images are also saved in `output_images` folder. The result video is in this repository. Following are explanation of those files.

* `camera_calibration.py` is used for calibrating camera and save result of `cv2.calibrateCamera()` to a pickle file
* `image_gen.py` is used for process all test image in the `test_image` folder. 
* `tracker.py` save some important parameters and has a function outputing an array of x coordinate of left and right center points for each level.
* `video_process.py` is used for process video

For detailed desciption of the project, please read Writeup.ipynb file
