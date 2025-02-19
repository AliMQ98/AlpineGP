import numpy as np

from Fitness import fitness
from PredictScoreFuncs import predict, score
import V_Vdot_Calculations, Functions

import yaml
import os
from deap import gp
from alpine.gp.gpsymbreg import GPSymbolicRegressor
import warnings

# Constants
g, l, b, m = 9.81, 0.5, 0.1, 0.15
A, B = -g / l, -b / m

# Define the numerical grid (Domain)
x1_vals = np.linspace(-3, 3, 200)
x2_vals = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

def main():
    yamlfile = "Lyapunov.yaml"
    filename = yamlfile
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float, float
        ],
        float,
    )
    pset.renameArguments(ARG0="x1", ARG1="x2")

    penalty = config_file_data["gp"]["penalty"]
    common_data = {"penalty": penalty}

    gpsr = GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=score.remote,
        predict_func=predict.remote,
        common_data=common_data,
        config_file_data=config_file_data,
        print_log=True,
        batch_size=100,
    )

    train_data = Functions.Dataset("true_data", [x1_vals, x2_vals], None)
    gpsr.fit(train_data)

    ray.shutdown()


if __name__ == "__main__":
    main()