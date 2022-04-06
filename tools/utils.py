import json

import numpy as np
import pandas as pd
import astropy.table


def to_numpy(outflow_properties):
    return (
        outflow_properties.as_array()
        .view(np.float64)
        .reshape((len(outflow_properties), -1))
    )


def from_numpy(data, column_names):
    return astropy.table.Table(data, names=column_names)


def split_sets_ids(outflow_properties, flex_point=0.8):
    rng = np.random.default_rng(0)
    all_ids = np.unique(outflow_properties["id"].data)
    rng.shuffle(all_ids)

    split_point = int(flex_point * len(all_ids))

    train_ids = all_ids[:split_point]
    test_ids = all_ids[split_point:]

    return train_ids, test_ids


def split_sets_masks(outflow_properties, flex_point=0.8):
    train_ids, test_ids = split_sets_ids(outflow_properties, flex_point=flex_point)
    train_mask = np.isin(outflow_properties["id"], train_ids)
    test_mask = np.isin(outflow_properties["id"], test_ids)

    return train_mask, test_mask


def load_real_outflows(path="./observed_outflows.csv"):
    real_outflows = pd.read_csv(path, index_col="name")
    real_outflows = real_outflows[real_outflows["type"] == "full"]

    real_outflows["mass_out"] = 10 ** real_outflows["mass_out_log"]
    real_outflows["smbh_mass"] = 10 ** real_outflows["smbh_mass_log"]
    real_outflows["luminosity_AGN"] = 10 ** real_outflows["luminosity_AGN_log"]
    real_outflows["dot_mass"] = real_outflows["derived_dot_mass"]

    return real_outflows


def load_simulated_outflows(path="./outputs/outflows.hdf5"):
    outflow_properties = astropy.table.Table.read(path)
    outflow_properties = outflow_properties[outflow_properties["dot_radius"] > 0]
    outflow_properties = outflow_properties[outflow_properties["luminosity_AGN"] > 0]

    return outflow_properties


output_params = [
    "duty_cycle",
    "quasar_activity_duration",
    "bulge_mass",
    "outflow_solid_angle_fraction",
    "bulge_gas_fraction",
]

input_params = [
    "radius",
    "dot_radius",
    "dot_mass",
    "mass_out",
    "smbh_mass",
    "luminosity_AGN",
]


def fit_normalization(X, y):
    X_mean = np.mean(np.log10(X), axis=0)
    X_stddev = np.std(np.log10(X), axis=0)

    y_mean = np.mean(np.log10(y), axis=0)
    y_stddev = np.std(np.log10(y), axis=0)

    np.savez(
        "outputs/normalization_parameters.npz",
        X_mean=X_mean,
        X_stddev=X_stddev,
        y_mean=y_mean,
        y_stddev=y_stddev,
    )

    return X_mean, X_stddev, y_mean, y_stddev


def load_normalization():
    data = np.load("outputs/normalization_parameters.npz")
    return data["X_mean"], data["X_stddev"], data["y_mean"], data["y_stddev"]


def normalize(data, mean, stddev):
    return (np.log10(data) - mean) / stddev


def denormalize(data, mean, stddev):
    return 10 ** (data * stddev + mean)
