# Tommy's MRI-Volume-Masker-3000-TM

Use to draw volumetric masks for brain data.

![example image](https://github.com/TommyClausner/MRI-Volume-Masker-3000-TM/blob/main/example.png?raw=true)

### start
Mask will be precomputed:

`python vol2mask.py /path/to/volume.nii.gz`

or to load a previously computed mask:

`python vol2mask.py /path/to/volume.nii.gz -m /path/to/m_volume.nii.gz`

or no mask will be used:

`python vol2mask.py /path/to/volume.nii.gz -m none`

Afterwards maximize figure window.

### install

`git clone https://github.com/TommyClausner/MRI-Volume-Masker-3000-TM/`

`cd MRI-Volume-Masker-3000-TM`

`pip install -r requirements.txt`
