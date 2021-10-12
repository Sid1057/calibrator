# Calibrator

An application that allows you to estimate the intrinsic and extrinsic calibration parameters of your camera using your hands and your knowledge of the scene :)

From

![](https://raw.githubusercontent.com/Sid1057/calibrator/main/docs/win.png)

To

![](https://raw.githubusercontent.com/Sid1057/calibrator/main/docs/result.png)

## Run

```sh
python3 calibrator.py --config conf/config.yaml --img docs/win.png
```

## How To Use

### Arguments

```
--config  # path to yaml configuration file
--img     # path to uncalibrated image
```

### HotKeys

Press 's' to save your calibration parameters

### Configuration

#### Input/Output section

starts in `io`

|Parameter|Description|
|-|-|
|output_dir|Directory to save your calibration parameters|

#### Interface Section

starts in `interface`

|Parameter|Description|
|-|-|
|show_pp|show principal point on image|
|render_cube|render virtual cube to estimate extrinsic parameters|
|show_ROI|show undistorted image after roi selecting|
|show_withour_ROI|show undistorted image before roi selecting|
|show_original|show original uncalibrated image|
|print_online|update actual parameters information in terminal|
|width|interface window width|

#### Calibration parameters section

In next section each value has 3 parameters:

|Parameter|Description|
|-|-|
|min_val| Minimum parameter value |
|max_val| Maximum parameter value |
|step| Slider step |

**!NB**: OpenCV provides a slider interface that have to start from zero and have only integer values. To see the real value of any parameters, see the output of the application in the terminal.

Sections:

- K
  - F  - limits of the focal length
  - CX - limits of the principal point
  - CY - limits of the principal point
- D - limits of the distorsion coefficients
  - D0
	- D1
	- D2
	- D3
  - D4
- T - limits of the translation vector
  - X
  - Y
  - Z
- R - limits of the rotation vector
  - yaw
  - pitch
  - roll

- CUBE - limits of the AR cube parameters
  - length
  - QX
  - QY
  - QZ

### Windows

![](https://raw.githubusercontent.com/Sid1057/calibrator/main/docs/screenshot.png)

|Window|Description|
|-|-|
|terminal|show you actual information about your estimated parameters|
|intrinsic|slide bar for intrinsic camera calibration parameters|
|extrinsic|slide bar for intrinsic camera calibration parameters|
|cube|slide bar for physical virtual cube parameters|
|original|original image|
|result|image after applying calibration parameters|
|result with ROI|image after applying calibration parameters and ROI selection|


### Output

|File|Description|
|-|-|
|K_original.txt|intrinsic camera parameters|
|D.txt|distorion coefficients|
|roi.txt|ROI for cutting unusable image area|
|K.txt|intrinsic camera parameters after roi selection|
|dist.txt|distorion coefficients|
|rvec.txt|rotation vector|
|tvec.txt|translation vector|
