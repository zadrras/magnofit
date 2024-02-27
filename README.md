# Magnofit

Magnofit is a library for running 1D outflow simulations of active galactic nuclei (AGNs).

<p align="center" float="left">
  <img src="figures/outflow_scatter_plot.png" alt="Simulated outflow parameters" style="height:440px"/>
  <img src="figures/real_predictions.png" alt="Simulated outflow parameters" style="height:440px"/>
</p>

[Poetry](https://python-poetry.org/) is used to install dependencies and manage the project's virtual environment. Magnofit also uses Python 3.11 which you'll need to install if your system comes with a different version. The preferred way to do this is via [pyenv](https://github.com/pyenv/pyenv).

## Quick setup guide

1. Install [Poetry](https://python-poetry.org/) and [pyenv](https://github.com/pyenv/pyenv).

2. Clone the repository and enter it's directory:

```bash
git clone git@github.com:zadrras/magnofit.git
cd magnofit
```

3. Install and activate Python 3.11 for the project:

```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

4. Create virtual environment and install dependencies:

```bash
poetry install
```

The virtual environment will be located inside the project directory and named `.venv`. You can use this through Poetry or by activating it manually.

5. Run tests to make sure the project is functioning as intended:

```bash
poetry run pytest
```

## Usage

To generate a sample outflow table run [tools/generate.py](tools/generate.py):

```bash
poetry run python tools/generate.py
```

This will output an Astropy table of outflows in `outputs/outflows.hdf5`. You can inspect it using Python:

```python
outflow_properties = astropy.table.Table.read("outputs/outflows.hdf5")
print(outflow_properties)
```

## Replicating the paper

Do note that to replicate the paper exactly you will need to checkout the commit tagged as [`paper`](https://github.com/zadrras/magnofit/releases/tag/paper). Newer versions of the code might produce slightly different outflows and figures.

### Steps

Generate outflow table with a varied range of parameters:

```bash
poetry run python tools/generate.py
```

This takes around 1 hour and 40 minutes on 16 cores of an AMD Ryzen 7 3800X CPU.

Train a neural network to predict the duty cycle, quasar activity duration, bulge mass, solid angle fraction and bulge gas fraction of the outflow:

```bash
poetry run python tools/train.py
```

This takes under 14 minutes on an AMD Ryzen 7 3800X processor.

Predict the parameters of real AGN outflows (found in [observed_outflows.csv](observed_outflows.csv)):

```bash
poetry run python tools/predict_real_data.py
```

Generate simulated outflows from the neural-network-predicted parameters of real AGN outflows:

```bash
poetry run python tools/generate_from_real_outflows.py
```

Plot figures and save them to [`figures/`](figures/):

```bash
poetry run ./tools/plot_all.sh
```

## Contributing
Pull requests are welcome. Please open an issue first to discuss what you would like to change.

## License
The project is made available under the MIT license. See the [LICENSE](LICENSE.md) file for more information. If you use this software when preparing a publication, please cite [Zubovas, Bialopetravičius & Kazlauskaitė (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220701959Z).
