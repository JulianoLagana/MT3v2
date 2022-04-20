"""
Script that helps tune a trained MT3v2 model's PPP intensity. The result path is specified via -rp argument,
and its predictions are computed on the fly by the script. Make sure to tune the PPP intensity using a different seed
than the one used for evaluation later.
"""


from util.misc import super_load, super_load_legacy
from util.load_config_files import load_yaml_into_dotdict
import argparse
import warnings


# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', help='filepath to result folder for trained model', required=True)
parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
parser.add_argument('--legacy', help='Flag to use legacy version of super_load or not', action='store_true', required=False)
parser.add_argument('--show_debugging_plots', help='Flag to decide if samples will be plotted or not', action='store_true', required=False)
parser.add_argument('--plot_decomposition_for_all_samples', action='store_true', required=False)
args = parser.parse_args()
print(f'Evaluating results from folder: {args.result_filepath}...')

if args.legacy:
    model, params = super_load_legacy(args.result_filepath, verbose=True)
else:
    model, params = super_load(args.result_filepath, verbose=True)

# Test that the model was trained in the task chosen for evaluation
if args.task_params is not None:
    task_params = load_yaml_into_dotdict(args.task_params)
    for k, v in task_params.data_generation.items():
        if k not in params.data_generation:
            warnings.warn(f"Key '{k}' not found in trained model's hyperparameters")
        elif params.data_generation[k] != v:
            warnings.warn(f"Different values for key '{k}'. Task: {v}\tTrained: {params.data_generation[k]}")
    # Use task params, not the ones from the trained model
    params.recursive_update(task_params)  # note: parameters specified only on trained model will remain untouched
else:
    warnings.warn('Evaluation task was not specified; inferring it from the task specified in the results folder.')


import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_generation.data_generator import DataGenerator
from modules.loss import MotLoss

from util.pmb_utils import compute_nll_for_pmb


# Read evaluation hyperparameters and overwrite `params` with them
eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)

# Check this script is running on correct settings
if params.data_generation.prediction_target != 'position_and_velocity':
    raise NotImplementedError('This script only works when models are predicting position and velocity estimates.')
if params.training.batch_size != 1:
    raise NotImplementedError('This script only works for batch size == 1')

mot_loss = MotLoss(params)

lambda_values = np.linspace(0.01, 0.2, 10)  # Change these while tuning
results = []
with torch.no_grad():
    for lambda_value in lambda_values:

        nll_results = {'total': [], 'loc': [], 'p_true': [], 'p_false': [], 'false': []}
        data_generator = DataGenerator(params)
        t = time.time()
        print(f"λ: {lambda_value}")

        for i in range(params.n_samples):

            # Get batch from data generator and feed it to trained model
            batch, labels, _, _, trajectories = data_generator.get_batch()
            prediction, _, _, _, _ = model.forward(batch)

            # Compute NLL
            predictions_mt3 = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
                               'logits': prediction.logits,
                               'state_covariances': prediction.uncertainties**2}
            predicted_pmb_mt3 = {'means': predictions_mt3['state'][0],
                                 'covs': prediction.uncertainties[0]**2,
                                 'p_exs': torch.sigmoid(prediction.logits[0]),
                                 'ppp_lambda': lambda_value,
                                 'ppp_log_prob_func': lambda x: np.log(lambda_value / data_generator.datagens[0].field_of_view.area())}
            nll_sample_mt3, nll_decomposition_mt3, _, _ = compute_nll_for_pmb(predicted_pmb_mt3, labels[0])

            # Save results from this sample
            nll_results['total'].append(nll_sample_mt3)
            nll_results['loc'].append(nll_decomposition_mt3[0])
            nll_results['p_true'].append(nll_decomposition_mt3[1])
            nll_results['p_false'].append(nll_decomposition_mt3[2])
            nll_results['false'].append(nll_decomposition_mt3[3])

            if (i % (1 + (params.n_samples // 10))) == 0:
                print(f"\tProcessed {i + 1}/{params.n_samples}...")

        average_nll = np.mean(nll_results['total'])
        print(f"\tNLL = {average_nll:.2f}. Finished in {time.time()-t:.2f} seconds.")
        results.append(average_nll)

    print(f"Plotting results")
    plt.plot(lambda_values, results, '-*')
    plt.show()

    print(results)
