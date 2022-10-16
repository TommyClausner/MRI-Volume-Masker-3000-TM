# Tommy's MRI-Volume-Masker-3000-TM 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7211758.svg)](https://doi.org/10.5281/zenodo.7211758)


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
binary mask view: b
switch draw mode: d
switch view plane: v
switch filter: f
toggle pan: p
toggle zoom: z
reset zoom: escape
export mask: e
quit: q
load file: o
load mask: ctrl+o
```

![example image](https://github.com/TommyClausner/MRI-Volume-Masker-3000-TM/blob/main/example.png?raw=true)

### start
Open file selection dialog and precompute mask for selected:

`python vol2mask.py`

Mask will be precomputed:

`python vol2mask.py -f /path/to/volume.nii.gz`

or to load a previously computed mask:

`python vol2mask.py -f /path/to/volume.nii.gz -m /path/to/m_volume.nii.gz`

or no mask will be used:

`python vol2mask.py -f /path/to/volume.nii.gz -m none`

Afterwards maximize figure window.

### install

`git clone https://github.com/TommyClausner/MRI-Volume-Masker-3000-TM/`

`cd MRI-Volume-Masker-3000-TM`

`pip install -r requirements.txt`

### matplotlib backend
GUI functions rely on matplotlib and thus the correct choice of a backend. The default is `WxAgg`. However on MacOS sometimes the GUI window is not showing up. The solution is to set a different backend in the `config.json` which will be created after the first start of the program, using `config_template.json` as a template. In `config.json` change the backend entry to something that works for you. See https://matplotlib.org/stable/tutorials/introductory/usage.html#what-is-a-backend

What I found works:
- Windows: WxAgg
- Linux: WxAgg
- MacOS: TkAgg or Qt5Agg

The reason why TkAgg is not the default, is that on Windows machines users experienced some flickering / jumping glitches that do not occur using WxAgg. However WxAgg has issues on MacOS. TkAgg seems to work fine for that.

For several backends (WxAgg / Qt5Agg / ...) third party packages need to be installed (Wx is already in the `requirements.txt`).

You can easily find out which additional package / software is needed for the respective backend using a web search engine of your choice.
