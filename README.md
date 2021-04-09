# Tommy's MRI-Volume-Masker-3000-TM

Use to draw volumetric masks for brain data. Requires python >= 3.7

```
################################################
Button mapping Tommy's MRI Volume Masker 3000 TM
################################################

set slice: enter
slice up: up
slice down: down
increase brightness: right
decrease brightness: left
increase mask alpha: +
decrease mask alpha: -
enable / disable mask: m
switch draw mode: d
switch view plane: v
switch filter: f
toggle pan: p
toggle zoom: z
reset zoom: escape
export mask: e
quit: q
new file: n
```

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
