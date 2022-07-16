#!/bin/bash

python tools/plot_outflow_scatter.py
python tools/predict.py
python tools/predict_errors.py
python tools/predict_real_data.py
python tools/predict_real_data_by_type.py
python tools/plot_real_outflows.py
python tools/plot_bout_vs_lkin.py
