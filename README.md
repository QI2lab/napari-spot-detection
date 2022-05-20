# napari-spot-detection

[![License](https://img.shields.io/pypi/l/napari-spot-detection.svg?color=green)](https://github.com/AlexCoul/napari-spot-detection/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-spot-detection.svg?color=green)](https://pypi.org/project/napari-spot-detection)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spot-detection.svg?color=green)](https://python.org)
[![tests](https://github.com/AlexCoul/napari-spot-detection/workflows/tests/badge.svg)](https://github.com/AlexCoul/napari-spot-detection/actions)
[![codecov](https://codecov.io/gh/AlexCoul/napari-spot-detection/branch/main/graph/badge.svg)](https://codecov.io/gh/AlexCoul/napari-spot-detection)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spot-detection)](https://napari-hub.org/plugins/napari-spot-detection)

Interactive parameters selection and visualization of intermediate results for spot detection.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

~~You can install `napari-spot-detection` via [pip]:~~
~~pip install napari-spot-detection~~
For now the package has complex dependencies, hence the lack of automated tests and CI.  
You'll need ideally a working CUDA installation (brace yourself), and non optionally, the custom versions of GPUfit as well as the localize-psf library:

```bash
# Prepare installation directory
INSTALL_DIR='~/Programs'  # where libraries will be downloaded and installed
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# activate your pyenv of conda python environment, then:

# Install GPUfit:
git clone https://github.com/QI2lab/Gpufit.git
cd Gpufit
# path to cmake command
cmake_path=$(which cmake)
# directory where gpu
build_dir="../gpufit_build"
mkdir $build_dir
cd $build_dir
$cmake_path -DCMAKE_BUILD_TYPE=RELEASE ../Gpufit
make
# go to the folder with compiled library pyGpufit
cd $build_dir/pyGpufit
pip install .

cd $INSTALL_DIR
# Install localize-psf:
git clone https://github.com/QI2lab/localize-psf.git
cd localize-psf
# in setup.py I changed `extras = {'gpu': ['cupy'],` to `extras = {'gpu': ['cupy-cuda114'],`
# to match my CUDA installation and avoid a ore recent (incompatible) version to be installed
pip install --no-cache-dir .['gpu']
# or pip install --no-cache-dir . if CUDA is not available

# Install napari-spot-detection:
cd $INSTALL_DIR
git clone https://github.com/AlexCoul/napari-spot-detection.git
cd napari-spot-detection
pip install .
# or pip install -e .  if you want to modify the plugin
```



To install latest development version :

    pip install git+https://github.com/AlexCoul/napari-spot-detection.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"napari-spot-detection" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/AlexCoul/napari-spot-detection/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
