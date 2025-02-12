# Simphony: A Simulator for Photonic circuits

<p align="center">
<img alt="Development version" src="https://img.shields.io/badge/master-v0.6.1-informational">
<a href="https://pypi.python.org/pypi/simphony"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/simphony.svg"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/simphony">
<a href="https://github.com/BYUCamachoLab/simphony/actions?query=workflow%3A%22build+%28pip%29%22"><img alt="Build Status" src="https://github.com/BYUCamachoLab/simphony/workflows/build%20(pip)/badge.svg"></a>
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit" style="max-width:100%;"></a>
<a href="https://simphonyphotonics.readthedocs.io/"><img alt="Documentation Status" src="https://readthedocs.org/projects/simphonyphotonics/badge/?version=latest"></a>
<a href="https://pypi.python.org/pypi/simphony/"><img alt="License" src="https://img.shields.io/pypi/l/simphony.svg"></a>
<a href="https://github.com/BYUCamachoLab/simphony/commits/master"><img alt="Latest Commit" src="https://img.shields.io/github/last-commit/BYUCamachoLab/simphony.svg"></a>
</p>

![](simphony_logo.png)

Simphony, a simulator for photonic circuits, is a fundamental package for designing and simulating photonic integrated circuits with Python.

**Key Features:**

- Free and open-source software provided under the MIT License
- Completely scriptable using Python 3.
- Cross-platform: runs on Windows, MacOS, and Linux.
- Subnetwork growth routines
- A simple, extensible framework for defining photonic component compact models.
- A SPICE-like method for defining photonic circuits.
- Complex simulation capabilities.
- Included model libraries from SiEPIC and SiPANN.

Developed by [CamachoLab](https://camacholab.byu.edu/) at
[Brigham Young University](https://www.byu.edu/).

## Installation

Simphony can be installed via pip using Python 3:

```
python3 -m pip install simphony
```

Please note that Python 2 is not supported. With the official deprecation of
Python 2 (January 1, 2020), no future compatability is planned.

## Documentation

The documentation is hosted [online](https://simphonyphotonics.readthedocs.io/en/latest/).

Changelogs can be found in docs/changelog/. There is a changelog file for
each released version of the software.

## Bibtex citation

```
@article{DBLP:journals/corr/abs-2009-05146,
  author    = {Sequoia Ploeg and
               Hyrum Gunther and
               Ryan M. Camacho},
  title     = {Simphony: An open-source photonic integrated circuit simulation framework},
  journal   = {CoRR},
  volume    = {abs/2009.05146},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.05146},
  eprinttype = {arXiv},
  eprint    = {2009.05146},
  timestamp = {Thu, 17 Sep 2020 12:49:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-05146.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
