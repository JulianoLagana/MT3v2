import os
import time
import datetime
import re
import shutil
from collections import deque
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_generation.data_generator import DataGenerator
from util.misc import save_checkpoint, update_logs
from util.load_config_files import load_yaml_into_dotdict
from util.plotting import output_truth_plot, compute_avg_certainty, get_constrastive_ax, get_false_ax, \
    get_total_loss_ax, get_state_uncertainties_ax
from util.logger import Logger
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules import evaluator
from modules.models.mt3v2.mt3v2 import MT3V2


if __name__ == '__main__':

    # Load CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--task_params', help='filepath to configuration yaml file defining the task', required=True)
    parser.add_argument('-mp', '--model_params', help='filepath to configuration yaml file defining the model', required=True)
    parser.add_argument('--continue_training_from', help='filepath to folder of an experiment to continue training from')
    parser.add_argument('--exp_name', help='Name to give to the results folder')
    args = parser.parse_args()
    print(f'Task configuration file: {args.task_params}')
    print(f'Model configuration file: {args.model_params}')

    # Load hyperparameters
    params = load_yaml_into_dotdict(args.task_params)
    params.update(load_yaml_into_dotdict(args.model_params))
    eval_params = load_yaml_into_dotdict(args.task_params)
    eval_params.update(load_yaml_into_dotdict(args.model_params))
    eval_params.recursive_update(load_yaml_into_dotdict('configs/eval/default.yaml'))
    eval_params.data_generation.seed += 1  # make sure we don't evaluate with same seed as final evaluation after training

    # Generate 32-bit random seed, or use user-specified one
    if params.general.pytorch_and_numpy_seed is None:
        random_data = os.urandom(4)
        params.general.pytorch_and_numpy_seed = int.from_bytes(random_data, byteorder="big")
    print(f'Using seed: {params.general.pytorch_and_numpy_seed}')

    # Seed pytorch and numpy for reproducibility
    torch.manual_seed(params.general.pytorch_and_numpy_seed)
    torch.cuda.manual_seed_all(params.general.pytorch_and_numpy_seed)
    np.random.seed(params.general.pytorch_and_numpy_seed)

    if params.training.device == 'auto':
        params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if eval_params.training.device == 'auto':
        eval_params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create logger and save all code dependencies imported so far
    cur_path = os.path.dirname(os.path.abspath(__file__))
    results_folder_path = cur_path + os.sep + 'results'
    exp_name = args.exp_name if args.exp_name is not None else time.strftime("%Y-%m-%d_%H%M%S")
    
    logger = Logger(log_path=f'{results_folder_path}/{exp_name}', save_output=False, buffer_size=params.debug.log_interval)
    print(f"Saving results to folder {logger.log_path}")
    logger.save_code_dependencies(project_root_path=os.path.realpath('../'))  # assuming this is ran from repo root
    logger.log_scalar('seed', params.general.pytorch_and_numpy_seed, 0, flush_now=True)

    # Manually copy the configuration yaml file used for this experiment to the logger folder
    shutil.copy(args.task_params, os.path.join(logger.log_path, 'code_used', 'task_params.yaml'))
    shutil.copy(args.model_params, os.path.join(logger.log_path, 'code_used', 'model_params.yaml'))

    # If continuing an experiment, manually copy the `code_used` of the experiment from which training wil continue
    if args.continue_training_from is not None:
        try:
            shutil.copytree(os.path.join(args.continue_training_from, 'code_used'),
                            os.path.join(logger.log_path, 'code_from_previous_training'))
        except FileNotFoundError:
            print(f'Path specified to continue training from does not exist: {args.continue_training_from}')
            exit()

    # Accumulate gradients to save memory
    n_splits = params.training.n_splits if params.training.n_splits is not None else 1
    if not (params.training.batch_size % n_splits == 0):
        raise ValueError("'params.training.batch_size' must be divdeble with 'params.training.n_splits'")
    params.training.batch_size = params.training.batch_size/n_splits

    model = MT3V2(params)

    # Create data generators for training
    data_generator = DataGenerator(params)

    # Create losses for training and evaluation
    mot_loss = MotLoss(params)
    mot_loss.to(params.training.device)
    contrastive_loss = ContrastiveLoss(params)
    mot_loss_eval = MotLoss(eval_params)
    mot_loss_eval.to(eval_params.training.device)

    # Optionally load the model weights from a provided checkpoint
    if args.continue_training_from is not None:
        # Find filename for last checkpoint available
        checkpoints_path = os.path.join(args.continue_training_from, 'checkpoints')
        checkpoint_names = os.listdir(checkpoints_path)
        checkpoint_names = [c for c in checkpoint_names if '.DS_Store' not in c]
        idx_last = np.argmax([int(re.findall(r"\d+", c)[-1]) for c in checkpoint_names])  # extract last occurrence of a number from the names
        last_filename = os.path.join(checkpoints_path, checkpoint_names[idx_last])

        # Load model weights and pass model to correct device
        checkpoint = torch.load(last_filename, map_location=params.training.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(torch.device(params.training.device))
    optimizer = AdamW(model.parameters(), lr=params.training.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=params.training.reduce_lr_patience,
                                  factor=params.training.reduce_lr_factor,
                                  verbose=params.debug.print_reduce_lr_messages)
    # Optionally load optimizer and scheduler states from provided checkpoint (this has to be done after loading the
    # model weights and calling model.to(), to guarantee these will be in the correct device too)
    if args.continue_training_from is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        del checkpoint

    current_lr = optimizer.param_groups[0]['lr']
    logger.log_scalar('metrics/learning_rate', current_lr, 0, flush_now=True)

    if params.debug.enable_plot or params.debug.save_plot_figs:
        fig = plt.figure(constrained_layout=True, figsize=(15, 8))
        fig.canvas.set_window_title('Training Progress')

        gs = GridSpec(2, 3, figure=fig)
        loss_ax = fig.add_subplot(gs[0, 0])
        loss_ax.set_ylabel('Loss', color='C0')
        loss_ax.grid('on')
        loss_line, = loss_ax.plot([1], 'r', label='Loss', c='C0')
        loss_ax.tick_params(axis='y', labelcolor='C0')

        percent_ax = fig.add_subplot(gs[1, 0])
        percent_ax.set_ylabel('Certainty distribution')
        percent_ax.grid('on')
        matched_median_cert_line, = percent_ax.plot([1], 'C0', label='Matched median certainty')
        unmatched_median_cert_line, = percent_ax.plot([1], 'C3', label='Unmatched median certainty')
        max_cert_line, = percent_ax.plot([1], 'C0--', label='Max certainty')
        min_cert_line, = percent_ax.plot([1], 'C0--', label='Min certainty')
        
        output_ax = fig.add_subplot(gs[:, 1:])
        output_ax.set_ylabel('Y')
        output_ax.set_xlabel('X')
        output_ax.set_aspect('equal', 'box')

        if params.debug.save_plot_figs:
            os.makedirs(os.path.join(logger.log_path, 'figs', 'main'))
            total_loss_fig, total_loss_ax, total_loss_line = get_total_loss_ax()
            os.makedirs(os.path.join(logger.log_path, 'figs', 'aux'))

        if params.loss.contrastive_classifier:
            contrastive_loss_fig, contrastive_loss_ax, contrastive_loss_line = get_constrastive_ax()
            os.makedirs(os.path.join(logger.log_path, 'figs', 'aux', 'contrastive'))
            
        if params.loss.false_classifier:
            false_loss_fig, false_loss_ax, false_loss_line = get_false_ax()
            os.makedirs(os.path.join(logger.log_path, 'figs', 'aux', 'false'))

        state_uncertainties_fig, state_uncertainties_ax, state_uncertainties_lines = get_state_uncertainties_ax()
        state_uncertainties_ax.legend()
        os.makedirs(os.path.join(logger.log_path, 'figs', 'aux', 'state_uncertainties'))

    losses = []
    last_layer_losses = []
    c_losses = []
    f_losses = []
    matched_min_certainties = []
    matched_q1_certainties = []
    matched_median_certainties = []
    matched_q3_certainties = []
    matched_max_certainties = []
    unmatched_min_certainties = []
    unmatched_q1_certainties = []
    unmatched_median_certainties = []
    unmatched_q3_certainties = []
    unmatched_max_certainties = []
    avg_stds_x = []
    avg_stds_y = []
    avg_stds_vx = []
    avg_stds_vy = []

    outputs_history = deque(maxlen=50)
    indices_history = deque(maxlen=50)

    print("[INFO] Training started...")
    start_time = time.time()
    time_since = time.time()

    for i_gradient_step in range(params.training.n_gradient_steps):
        logs = {}
        for i_split_step in range(n_splits):
            try:
                batch, labels, unique_ids, _, trajectories = data_generator.get_batch()
                prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = \
                    model.forward(batch)

                loss_dict, indices = mot_loss.forward(labels, prediction, intermediate_predictions, encoder_prediction, loss_type=params.loss.type)

                total_loss = sum(loss_dict[k] for k in loss_dict.keys())
                logs = update_logs(logs, 'last_layer_losses', loss_dict[f'{params.loss.type}_logits'].item() + loss_dict[f'{params.loss.type}_state'].item())

                if params.loss.contrastive_classifier:
                    c_loss = contrastive_loss(aux_classifications['contrastive_classifications'], unique_ids)
                    total_loss = total_loss + c_loss * params.loss.c_loss_multiplier
                    logs = update_logs(logs, 'contrastive_loss', c_loss.item())

                logs = update_logs(logs, 'total_loss', total_loss.item())
                
                if params.loss.return_intermediate:
                    for k, v in loss_dict.items():
                        if '_' in k or params.loss.type == 'dhn':
                            logs = update_logs(logs, k, v.item())

                # Compute quantiles for matched and unmatched predictions
                outputs_history.append({'logits': prediction.logits.detach().cpu()})
                indices_history.append(indices)
                matched_quants, unmatched_quants = compute_avg_certainty(outputs_history, indices_history)
                min_cert, q1_cert, median_cert, q3_cert, max_cert = matched_quants

                logs = update_logs(logs,'matched_min_certainty', min_cert)
                logs = update_logs(logs,'matched_q1_certainty', q1_cert)
                logs = update_logs(logs,'matched_median_certainty', median_cert)
                logs = update_logs(logs,'matched_q3_certainty', q3_cert)
                logs = update_logs(logs,'matched_max_certainty', max_cert)

                min_cert, q1_cert, median_cert, q3_cert, max_cert = unmatched_quants

                logs = update_logs(logs,'unmatched_min_certainty', min_cert)
                logs = update_logs(logs,'unmatched_q1_certainty', q1_cert)
                logs = update_logs(logs,'unmatched_median_certainty', median_cert)
                logs = update_logs(logs,'unmatched_q3_certainty', q3_cert)
                logs = update_logs(logs,'unmatched_max_certainty', max_cert)

                # Compute average standard deviation in state predictions
                avg_stds = prediction.uncertainties.mean(dim=0).mean(dim=0)
                logs = update_logs(logs, 'avg_std_x', avg_stds[0].item())
                logs = update_logs(logs, 'avg_std_y', avg_stds[1].item())
                logs = update_logs(logs, 'avg_std_vx', avg_stds[2].item())
                logs = update_logs(logs, 'avg_std_vy', avg_stds[3].item())

                total_loss.backward()

                all_model_weights = [p for p in list(model.parameters())]
                all_model_grads = [p.grad for p in list(model.parameters())]
                #pickle.dump(all_model_weights, open(f'weights_iter{i_gradient_step}.p', 'wb'))
                #pickle.dump(all_model_grads, open(f'grads_iter{i_gradient_step}.p', 'wb'))

                #def compare_but_exclude_nones(list_a, list_b):
                #    results = []
                #    for a, b in zip(list_a, list_b):
                #        if a is None or b is None:
                #            if a is None and b is None:
                #                results.append(True)
                #            else:
                #                results.append(False)
                #        elif torch.allclose(a, b, atol=1e-04, rtol=1e-02):
                #            results.append(True)
                #        else:
                #            results.append(False)
                #    return results
                #old_weights = pickle.load(open(f'weights_iter{i_gradient_step}.p', 'rb'))
                #old_grads = pickle.load(open(f'grads_iter{i_gradient_step}.p', 'rb'))
                #results_weights = compare_but_exclude_nones(all_model_weights, old_weights)
                #results_grads = compare_but_exclude_nones(all_model_grads, old_grads)

                if params.training.max_gradnorm is not None and params.training.max_gradnorm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params.training.max_gradnorm)

            except KeyboardInterrupt:
                filename = f'checkpoint_gradient_step_{i_gradient_step}'
                folder_name = os.path.join(logger.log_path, 'checkpoints')
                save_checkpoint(folder=folder_name,
                                filename=filename,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler)
                print("[INFO] Exiting...")
                data_generator.pool.close()
                data_generator_eval.pool.close()
                exit()
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    print(f'Shape of measurements was {batch.tensors.shape}.')
                    optimizer.zero_grad()
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # Add the mean of entire batch to plot
        losses.append(np.mean(np.array(logs['total_loss'])))
        last_layer_losses.append(np.mean(np.array(logs['last_layer_losses'])))

        matched_min_certainties.append(np.mean(np.array(logs['matched_min_certainty'])))
        matched_q1_certainties.append(np.mean(np.array(logs['matched_q1_certainty'])))
        matched_median_certainties.append(np.mean(np.array(logs['matched_median_certainty'])))
        matched_q3_certainties.append(np.mean(np.array(logs['matched_q3_certainty'])))
        matched_max_certainties.append(np.mean(np.array(logs['matched_max_certainty'])))

        unmatched_min_certainties.append(np.mean(np.array(logs['unmatched_min_certainty'])))
        unmatched_q1_certainties.append(np.mean(np.array(logs['unmatched_q1_certainty'])))
        unmatched_median_certainties.append(np.mean(np.array(logs['unmatched_median_certainty'])))
        unmatched_q3_certainties.append(np.mean(np.array(logs['unmatched_q3_certainty'])))
        unmatched_max_certainties.append(np.mean(np.array(logs['unmatched_max_certainty'])))

        avg_stds_x.append(logs['avg_std_x'])
        avg_stds_y.append(logs['avg_std_y'])
        avg_stds_vx.append(logs['avg_std_vx'])
        avg_stds_vy.append(logs['avg_std_vy'])

        if params.loss.contrastive_classifier:
            c_losses.append(np.mean(np.array(logs['contrastive_loss'])))
        if params.loss.false_classifier:
            f_losses.append(np.mean(np.array(logs['false_loss'])))

        if i_gradient_step % params.debug.print_interval == 0:
            cur_time = time.time()
            t = str(datetime.timedelta(seconds=round(cur_time - time_since)))
            t_tot = str(datetime.timedelta(seconds=round(cur_time - start_time)))
            print(f"Number of gradient steps: {i_gradient_step + 1} \t "
                f"Loss: {np.mean(losses[-15:]):.3f} \t "
                f"Time per step: {(cur_time-time_since)/params.debug.print_interval:.2f}s \t "
                f"Total time elapsed: {t_tot}")
            time_since = time.time()

        if (params.debug.enable_plot and i_gradient_step % params.debug.plot_interval == 0) or \
                (params.debug.save_plot_figs and i_gradient_step % params.debug.save_plot_figs_interval == 0):
            x_axis = list(range(i_gradient_step+1))
            loss_line.set_data(x_axis, last_layer_losses)
            loss_ax.relim()
            loss_ax.autoscale_view()

            percent_ax.collections.clear()
            matched_median_cert_line.set_data(x_axis, np.array(matched_median_certainties))
            percent_ax.fill_between(x_axis, matched_min_certainties, matched_max_certainties, color='C0', alpha=0.3, linewidth=0.0)
            percent_ax.fill_between(x_axis, matched_q1_certainties, matched_q3_certainties, color='C0', alpha=0.6, linewidth=0.0)
            unmatched_median_cert_line.set_data(x_axis, np.array(unmatched_median_certainties))
            percent_ax.fill_between(x_axis, unmatched_min_certainties, unmatched_max_certainties, color='C3', alpha=0.3, linewidth=0.0)
            percent_ax.fill_between(x_axis, unmatched_q1_certainties, unmatched_q3_certainties, color='C3', alpha=0.6, linewidth=0.0)
            percent_ax.set_ylim([-0.05, 1.05])

            output_ax.cla()
            output_ax.grid('on')
            output_truth_plot(output_ax, prediction, labels, indices, batch, params)
            output_ax.set_xlim([-params.data_generation.field_of_view.max_range*0.2, params.data_generation.field_of_view.max_range*1.2])
            output_ax.set_ylim([-params.data_generation.field_of_view.max_range, params.data_generation.field_of_view.max_range])

            if params.loss.contrastive_classifier:
                contrastive_loss_line.set_data(x_axis, c_losses)
                contrastive_loss_ax.relim()
                contrastive_loss_ax.autoscale_view()

            if params.loss.false_classifier:
                false_loss_line.set_data(x_axis, f_losses)
                false_loss_ax.relim()
                false_loss_ax.autoscale_view()

            for line, data in zip(state_uncertainties_lines, [avg_stds_x, avg_stds_y, avg_stds_vx, avg_stds_vy]):
                line.set_data(x_axis, data)
            state_uncertainties_ax.relim()
            state_uncertainties_ax.autoscale_view()

            if (params.debug.enable_plot and i_gradient_step % params.debug.plot_interval == 0):
                fig.canvas.draw()
                plt.pause(0.01)

            if params.debug.save_plot_figs and i_gradient_step % params.debug.save_plot_figs_interval == 0:
                filename = f"gradient_step{i_gradient_step}.jpg"
                fig.savefig(os.path.join(logger.log_path, 'figs', 'main', filename))

                total_loss_line.set_data(x_axis, losses)
                total_loss_ax.relim()
                total_loss_ax.autoscale_view()
                total_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', filename))

                if params.loss.contrastive_classifier:
                    contrastive_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', 'contrastive', filename))
                if params.loss.false_classifier:
                    false_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', 'false', filename))
                state_uncertainties_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', 'state_uncertainties', filename))

        # Log all metrics 
        for k,v in logs.items():
            logger.log_scalar(os.path.join('metrics', k), np.mean(np.array(v)), i_gradient_step)

        # Do the gradient step
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate, logging it if changed
        scheduler.step(np.mean(np.array(logs['total_loss'])))
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            current_lr = new_lr
            logger.log_scalar('metrics/learning_rate', current_lr, i_gradient_step, flush_now=True)

        # Save checkpoint
        if (i_gradient_step+1) % params.training.checkpoint_interval == 0:
            filename = f'checkpoint_gradient_step_{i_gradient_step}'
            folder_name = os.path.join(logger.log_path, 'checkpoints')
            save_checkpoint(folder=folder_name,
                            filename=filename,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler)

        # Periodically evaluate model
        if params.debug.evaluate_gospa_interval is not None and \
                (i_gradient_step+1) % params.debug.evaluate_gospa_interval == 0:
            data_generator_eval = DataGenerator(eval_params)
            print("Starting periodic evaluation...")
            gospa_results = evaluator.evaluate_gospa(data_generator_eval, model, eval_params)
            logger.log_scalar('metrics/gospa_total', gospa_results[0], i_gradient_step, flush_now=True)
            logger.log_scalar('metrics/gospa_localization_error', gospa_results[1], i_gradient_step, flush_now=True)
            logger.log_scalar('metrics/gospa_localization_error_normalized', gospa_results[2], i_gradient_step, flush_now=True)
            logger.log_scalar('metrics/gospa_missed', gospa_results[3], i_gradient_step, flush_now=True)
            logger.log_scalar('metrics/gospa_false', gospa_results[4], i_gradient_step, flush_now=True)
            print("Done. Resuming training.")
