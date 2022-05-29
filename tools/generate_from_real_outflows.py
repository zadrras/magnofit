"""
Brief description

This script simulates an outflow for a specified number of galaxies with
randomised parameters, outputs the results as astropy tables, and saves
them to a hdf5 archive.
"""
import os
import time
import multiprocessing
import functools
from tqdm import tqdm

import numpy as np
import pandas as pd

from magnofit.galaxy import Galaxy
import magnofit.constants as const
from magnofit.simulation import run_outflow_simulation
import magnofit.calc.luminosity


def generate_model_parameters(predicted_outflows, inital_real_outflows):
    galaxy_param_collection = []
    rng = np.random.default_rng(0)
    for i in range(len(predicted_outflows)):
        galaxy_params = Galaxy(
            virial_mass=None,
            bulge_gas_fraction=predicted_outflows.bulge_gas_fraction[i],
            outflow_sphere_angle_ratio=predicted_outflows.outflow_solid_angle_fraction[i],
            duty_cycle=predicted_outflows.duty_cycle[i],
            quasar_activity_duration=predicted_outflows.quasar_activity_duration[i]
            / const.UNIT_YEAR,
            fade=magnofit.calc.luminosity.LuminosityFadeKing(),
            bulge_mass=predicted_outflows.bulge_mass[i] / const.UNIT_MSUN,
            smbh_mass=10 ** inital_real_outflows.smbh_mass_log[i] / const.UNIT_MSUN,
            name=predicted_outflows.name[i],
        )

        galaxy_params.generate_stochastic_parameters(rng)
        galaxy_param_collection.append(galaxy_params)

    return galaxy_param_collection


initial_real_outflows = pd.read_csv("observed_outflows.csv")
initial_real_outflows = initial_real_outflows[initial_real_outflows.type == "full"]
initial_real_outflows.reset_index(drop=True, inplace=True)

predicted_real_outflows = pd.read_csv("./outputs/real_predictions_lumpress2.csv")
generated_model_params = generate_model_parameters(
    predicted_real_outflows, initial_real_outflows
)

print()
print(f"Running simulations...")
start_time = time.time()
with multiprocessing.Pool(processes=16) as pool:
    outflow_properties_collection = list(
        tqdm(
            pool.imap(
                functools.partial(run_outflow_simulation, rng=None),
                generated_model_params,
            ),
            total=int(len(predicted_real_outflows)),
        )
    )
end_time = time.time()
print(f"Simulations took {end_time - start_time:.2f} s.")

print()
print(f"Joining and stacking tables...")
start_time = time.time()
outflow_dataframe = []
for galaxy_params, outflow_properties in zip(
    generated_model_params, outflow_properties_collection
):
    if outflow_properties is not None:
        galaxy_params = galaxy_params.to_table().to_pandas(index=True)
        outflow_properties = outflow_properties.to_pandas()
        outflow = outflow_properties.merge(galaxy_params, how="cross")
        outflow_dataframe.append(outflow)

for idx, outflow_properties in enumerate(outflow_dataframe):
    outflow_properties["id"] = idx
outflow_dataframe = pd.concat(outflow_dataframe, sort=False)
end_time = time.time()
print(f"Joining and stacking tables took {end_time - start_time:.2f} s.")

print()
print(f"Saving simulations to disk...")
start_time = time.time()
os.makedirs("./outputs", exist_ok=True)
outflow_dataframe.to_csv("./outputs/generated_outflows_from_real_predictions_lumpress2.csv")
end_time = time.time()
print(f"Saving took {end_time - start_time:.2f} s.")
