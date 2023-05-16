# Estimate the bias and variance of a variety of models.
from omegaconf.errors import ConfigAttributeError
import hydra
import numpy as np
from interpensembles.predictions import Model

import os

here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
# plt.style.use(os.path.join(here, "../etc/config/stylesheet.mplstyle"))
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
from itertools import combinations

# get bias, variance, and performance to plot
def get_arrays_toplot(models):
  """
  Takes as argument a dictionary of models:
  keys giving model names, values are dictionaries with paths to individual entries.
  :param models: names of individual models.
  """
  all_metrics = []
  for modelname, model in models.items():

    # for each model type there may be multiple entries:
    for i, (m, l) in enumerate(zip(model.filepaths, model.labelpaths)):
      # register a single model
      model = Model(m, "ind")
      filename = os.path.join(here, m)
      labelpath = os.path.join(here, l)
      model.register(filename=m,
                   inputtype=None,
                   labelpath=l,
                   logits=True,)

      acc, nll, brier, qunc \
        = model.get_accuracy(), model.get_nll(), model.get_brier(), model.get_qunc()
      print("{}: Acc: {:.3f}, NLL: {:.3f}, Brier: {:.3f} Qunc:{:.3f}".format(
        modelname, acc, nll, brier, qunc))
      all_metrics.append([modelname, m, l, acc, nll, brier, qunc])

  df = pd.DataFrame(all_metrics, columns=["models", "filepaths", "labelpaths", "acc", "nll", "brier", "qunc"])
  return df


@hydra.main(config_path="script_configs/datasets/imagenet", config_name="imagenet")
def main(args):
  # Set up results directory
  results_dir = Path(here) / "results/model_performance/{}.csv".format(args.title)
  os.makedirs(os.path.dirname(results_dir), exist_ok=True)

  print('\n dataset {} results_dir: {}\n'.format(args.title, results_dir))

  # Get performance metrics for each ensemble:
  df = get_arrays_toplot(args.models)
  # Dump to csv
  df.to_csv(str(results_dir))

  print('Stored performance of {} in {}'.format(args.title, results_dir))

  return


if __name__ == "__main__":
  main()
