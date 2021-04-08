# Tommy's MRI-Volume-Masker-3000-TM

Use to draw volumetric masks for brain data.

- draw mask using lasso selection
- use enter to set selection
- used d to switch between draw and remove mode
- use up / down to iterate through slices
- use v to switch view
- use +/- and left/right to adjust mask alpha adn brightness
- use f to apply different image filters
- use e to export the mask (m_original_file_name.nii.gz)
- use q to quit program

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
