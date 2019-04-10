"""Eval trained AEM on UCI + BSDS300 datasets."""
import tensorflow as tf
import numpy as np
import argparse
import json
import os

from utils.data_utils import UCI
from utils.aem import AEM
from dotmap import DotMap


def parse_args():
    parser = argparse.ArgumentParser()
    # Data I/O
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of model. Used to name summary directories.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of UCI dataset to use"
    )
    parser.add_argument(
        "--split", type=str, required=True, help="train / val / test split"
    )
    # Eval options
    parser.add_argument(
        "--n_importance_samples",
        type=int,
        default=20000,
        help="Number of importance samples used to estimate norm constant",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size during training"
    )
    # reproducibility
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed to use")
    return parser.parse_args()


def eval_model(eval_args):

    # Random seeds
    np.random.seed(eval_args.seed)
    tf.set_random_seed(eval_args.seed)

    # Restore options from summary dir
    summary_dir = "logs/UCI/{}/{}".format(eval_args.dataset, eval_args.model_name)
    with open("{}/args.json".format(summary_dir), "r") as file:
        args = DotMap(json.load(file))

    # Save eval args
    with open("{}/eval_args.json".format(summary_dir), "w") as file:
        json.dump(vars(eval_args), file)

    # Load data
    x_batch, data_val, data_test = UCI(args.dataset, args.batch_size)

    # Create dropout placeholders
    dropout_p_MADE = tf.placeholder_with_default(0.0, ())
    dropout_p_energy_net = tf.placeholder_with_default(0.0, ())

    # Build AEM
    aem = AEM(
        x_batch,
        n_importance_samples=eval_args.n_importance_samples,
        proposal_comp_scale_min=args.proposal_comp_scale_min,
        n_proposal_mixture_comps=args.n_proposal_mixture_comps,
        n_res_blocks_MADE=args.n_res_blocks_MADE,
        n_hidden_units_MADE=args.n_hidden_units_MADE,
        activation_MADE=args.activation_MADE,
        dropout_p_MADE=dropout_p_MADE,
        n_context_units_energy_net=args.n_context_units_energy_net,
        n_res_blocks_energy_net=args.n_res_blocks_energy_net,
        n_hidden_units_energy_net=args.n_hidden_units_energy_net,
        activation_energy_net=args.activation_energy_net,
        dropout_p_energy_net=dropout_p_energy_net,
    )

    # Saver for restore
    saver = tf.train.Saver()
    checkpoint_dir = "checkpoints/UCI/{}/{}".format(
        eval_args.dataset, eval_args.model_name
    )

    # Create output file
    file = open("{}/eval_{}_set.txt".format(summary_dir, eval_args.split), "w")
    print("{} evaluation".format(eval_args.dataset), file=file)
    print("=======================\n", file=file)

    # Get eval data
    if eval_args.mode == "val":
        data_eval = data_val
    elif eval_args.mode == "test":
        data_eval = data_test

    # Importance sampling eval loop
    print("Evaluating model...")
    with tf.Session() as sess:
        ckpt_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, ckpt_file)
        log_prob_est_all_ex = []
        log_prob_proposal_all_ex = []

        # Loop over batches of eval data
        n_eval_ex = data_eval.shape[0]
        n_eval_batches = np.ceil(n_eval_ex / eval_args.batch_size).astype("int")
        for b in range(n_eval_batches):
            print("Batch {} of {}".format(b + 1, n_eval_batches))
            x_batch_curr = data_eval[
                b * eval_args.batch_size : (b + 1) * eval_args.batch_size
            ]
            energy_context_curr, proposal_params_curr = sess.run(
                (aem.energy_context, aem.proposal_params), {x_batch: x_batch_curr}
            )
            for energy_context_ex, proposal_params_ex, x_batch_ex in zip(
                energy_context_curr, proposal_params_curr, x_batch_curr
            ):
                log_prob_est_ex, log_prob_proposal_ex = sess.run(
                    (aem.log_prob_est_data, aem.proposal_log_prob_data),
                    {
                        aem.energy_context: energy_context_ex[None, ...],
                        aem.proposal_params: proposal_params_ex[None, ...],
                        x_batch: x_batch_ex[None, ...],
                    },
                )
                log_prob_est_all_ex.append(log_prob_est_ex)
                log_prob_proposal_all_ex.append(log_prob_proposal_ex)
        log_prob_est_all = np.concatenate(tuple(log_prob_est_all_ex))
        log_prob_proposal_all = np.concatenate(tuple(log_prob_proposal_all_ex))

    # Compute mean, standard dev and standard error of log prob estimates
    log_prob_est_mean, log_prob_est_std = (
        np.mean(log_prob_est_all),
        np.std(log_prob_est_all),
    )
    log_prob_est_sterr = log_prob_est_std / np.sqrt(n_eval_ex)

    # Compute mean, standard dev and standard error of proposal log probs
    log_prob_proposal_mean, log_prob_proposal_std = (
        np.mean(log_prob_proposal_all),
        np.std(log_prob_proposal_all),
    )
    log_prob_proposal_sterr = log_prob_proposal_std / np.sqrt(n_eval_ex)

    # Save outputs
    print(
        "Importance sampling estimate with {} samples:".format(
            eval_args.n_importance_samples
        ),
        file=file,
    )
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean: {}".format(log_prob_est_mean), file=file)
    print("Stddev: {}".format(log_prob_est_std), file=file)
    print("Stderr: {}\n".format(log_prob_est_sterr), file=file)

    print("Proposal log probabilities:", file=file)
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean: {}".format(log_prob_proposal_mean), file=file)
    print("Stddev: {}".format(log_prob_proposal_std), file=file)
    print("Stderr: {}\n".format(log_prob_proposal_sterr), file=file)

    file.close()


if __name__ == "__main__":
    args = parse_args()
    eval_model(args)

