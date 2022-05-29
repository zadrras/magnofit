"""
Brief description

This script simulates an outflow for a specified number of galaxies with
randomised parameters, outputs the results as astropy tables, and saves
them to a hdf5 archive.
"""
import os
import time
import multiprocessing
from tqdm import tqdm

import astropy.table
import numpy as np
import pandas as pd

from magnofit.galaxy import Galaxy
import magnofit.constants as const
from magnofit.simulation import run_outflow_simulation
import magnofit.calc.luminosity
import magnofit.calc.mass


def generate_initial_parameter_collection_randomised(number=1):
    rng = np.random.default_rng(0)
    galaxy_param_collection = []
    for _ in range(number):
        galaxy_params = Galaxy(
            virial_mass=(10 ** rng.uniform(12, 14)) / const.UNIT_MSUN,
            bulge_gas_fraction=rng.uniform(0.001, 0.3),
            outflow_sphere_angle_ratio=rng.uniform(0.05, 1),
            duty_cycle=rng.uniform(0.04, 1),
            quasar_activity_duration=rng.uniform(10 ** 4.0, 10 ** 5.5)
            / const.UNIT_YEAR,
            fade=magnofit.calc.luminosity.LuminosityFadeKing(),
            #bulge_profile=magnofit.calc.mass.MassAlpha(rng.uniform(1.9, 2.2)),
        )
        galaxy_params.generate_stochastic_parameters(rng)
        galaxy_param_collection.append(galaxy_params)

    return galaxy_param_collection


if __name__ == "__main__":

    # How many galaxies would you like to generate and simulate outflows for?
    galaxy_collection_size = 50_000

    print("Seeding initial galaxy parameters...")
    galaxy_param_collection = generate_initial_parameter_collection_randomised(
        number=galaxy_collection_size
    )
    print(f"Generated {len(galaxy_param_collection)} initial galaxy parameter sets.")

    print()
    print(f"Running simulations...")
    start_time = time.time()

    def worker_wrapper(arg):
        args, kwargs = arg
        return run_outflow_simulation(args, **kwargs)

    with multiprocessing.Pool(processes=16) as pool:
        outflow_properties_collection = list(
            tqdm(
                pool.imap(
                    worker_wrapper,
                    [
                        (
                            g,
                            {
                                "rng": np.random.default_rng(i + 1),
                                "output_array_length": 200,
                            },
                        )
                        for i, g in enumerate(galaxy_param_collection)
                    ],
                ),
                total=galaxy_collection_size,
            )
        )
    end_time = time.time()
    print(f"Simulations took {end_time - start_time:.2f} s.")

    print()
    print(f"Joining and stacking tables...")
    start_time = time.time()
    outflow_dataframe = []
    negatives = 0
    small_rads = 0
    others = 0
    alls = 0
    big_but_negatives = 0
    for galaxy_params, outflow_properties in zip(
        galaxy_param_collection, outflow_properties_collection
    ):
        if outflow_properties == 1:
            negatives += 1
        elif outflow_properties == 2:
            small_rads += 1
        elif outflow_properties == 3:
            big_but_negatives += 1
        else:
            others += 1
        alls += 1
        if outflow_properties not in [1, 2, 3]:
            galaxy_params = galaxy_params.to_table().to_pandas()
            outflow_properties = outflow_properties.to_pandas()

            outflow = outflow_properties.merge(galaxy_params, how="cross")
            outflow_dataframe.append(outflow)
    print("Negatives", negatives/alls, "Small rads", small_rads/alls, "big but negative", big_but_negatives/alls)
    for idx, outflow_properties in enumerate(outflow_dataframe):
        outflow_properties["id"] = idx
    outflow_dataframe = pd.concat(outflow_dataframe, ignore_index=True, sort=False)
    outflow_table = astropy.table.Table.from_pandas(outflow_dataframe)
    end_time = time.time()
    print(f"Joining and stacking tables took {end_time - start_time:.2f} s.")
    print()
    print(f"Saving simulations to disk...")
    start_time = time.time()
    os.makedirs("./outputs", exist_ok=True)
    outflow_table.write(
        "./outputs/outflows_lumpress3.hdf5",
        format="hdf5",
        path="outflow_properties",
        serialize_meta=True,
        overwrite=True,
    )
    end_time = time.time()
    print(f"Saving took {end_time - start_time:.2f} s.")
    print(len(outflow_table))