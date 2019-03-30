""" 
Train AEM on UCI datasets
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

from utils.aem import AEM
from utils.data_utils import Datasets2D


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
        "--dataset", type=str, required=True, help="Name of 2D dataset to use"
    )
    parser.add_argument("--save_summaries", type=int, default=1, help="Save summaries")
    parser.add_argument("--save_plots", type=int, default=1, help="Save plots")
    parser.add_argument(
        "--save_checkpoints", type=int, default=0, help="Save checkpoints"
    )
    parser.add_argument(
        "--summary_interval", type=int, default=1000, help="Summary save interval"
    )
    # MADE options
    parser.add_argument(
        "--activation_MADE", type=str, default="relu", help="Activation for MADE"
    )
    parser.add_argument(
        "--n_res_blocks_MADE",
        type=int,
        default=4,
        help="Number of residual blocks in MADE",
    )
    parser.add_argument(
        "--n_hidden_units_MADE",
        type=int,
        default=256,
        help="Number of hidden units for res blocks in MADE",
    )
    parser.add_argument(
        "--dropout_p_MADE", type=float, default=0.0, help="Dropout probability in MADE"
    )
    # Energy net options
    parser.add_argument(
        "--activation_energy_net",
        type=str,
        default="relu",
        help="Activation for energy net",
    )
    parser.add_argument(
        "--n_res_blocks_energy_net",
        type=int,
        default=4,
        help="Number of residual blocks in energy net",
    )
    parser.add_argument(
        "--n_hidden_units_energy_net",
        type=int,
        default=128,
        help="Number of hidden units for res blocks in energy net",
    )
    parser.add_argument(
        "--n_context_units_energy_net",
        type=int,
        default=64,
        help="Number of units to condition on in energy nets",
    )
    parser.add_argument(
        "--dropout_p_energy_net",
        type=float,
        default=0.0,
        help="Dropout probability in energy nets",
    )
    # Proposal dist options
    parser.add_argument(
        "--n_proposal_mixture_comps",
        type=int,
        default=10,
        help="Number of mixture components in proposal dist",
    )
    parser.add_argument(
        "--proposal_comp_scale_min",
        type=float,
        default=1e-3,
        help="Min val for scale in proposal mixture components",
    )
    # Optimization
    parser.add_argument(
        "--alpha_warm_up_steps",
        type=int,
        default=5000,
        help="No. steps to train proposal before energy model ",
    )
    parser.add_argument(
        "--n_importance_samples",
        type=int,
        default=20,
        help="Number of importance samples used to estimate norm constant",
    )
    parser.add_argument(
        "--learning_rate_start",
        type=float,
        default=5e-4,
        help="Cosine annealing learning rate starting value",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size during training"
    )
    parser.add_argument(
        "--max_steps", type=int, default=400000, help="Total number of training steps"
    )
    # reproducibility
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed to use")
    return parser.parse_args()


def train_model(args):

    # Random seeds
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Load data
    x_batch, data = Datasets2D(args.dataset, batch_size=args.batch_size)

    # Build AEM
    aem = AEM(
        x_batch,
        n_importance_samples=args.n_importance_samples,
        proposal_comp_scale_min=args.proposal_comp_scale_min,
        n_proposal_mixture_comps=args.n_proposal_mixture_comps,
        n_res_blocks_MADE=args.n_res_blocks_MADE,
        n_hidden_units_MADE=args.n_hidden_units_MADE,
        activation_MADE=args.activation_MADE,
        dropout_p_MADE=args.dropout_p_MADE,
        n_context_units_energy_net=args.n_context_units_energy_net,
        n_res_blocks_energy_net=args.n_res_blocks_energy_net,
        n_hidden_units_energy_net=args.n_hidden_units_energy_net,
        activation_energy_net=args.activation_energy_net,
        dropout_p_energy_net=args.dropout_p_energy_net,
    )

    # Get log prob of data under model and proposal
    mean_log_prob_est_data = tf.reduce_mean(aem.log_prob_est_data)
    mean_proposal_log_prob_data = tf.reduce_mean(aem.proposal_log_prob_data)

    # Loss is alpha-weighted sum of log prob of data under energy model and proposal
    global_step = tf.Variable(0, trainable=False)
    if args.alpha_warm_up_steps > 0:
        alpha = tf.cast(global_step > args.alpha_warm_up_steps, tf.float32)
    else:
        alpha = tf.constant(1.0)
    loss = -(alpha * mean_log_prob_est_data + mean_proposal_log_prob_data)

    # Optimization
    learning_rate = tf.train.cosine_decay(
        args.learning_rate_start, global_step, args.max_steps, name=None
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optim_op = optimizer.minimize(loss, global_step=global_step)

    # Summaries
    summary_dict = {
        "norm_constants_est": tf.reduce_mean(aem.norm_constants_est),
        "model_log_prob_est": mean_log_prob_est_data,
        "proposal_log_prob": mean_proposal_log_prob_data,
        "learning_rate": learning_rate,
        "alpha": alpha,
    }
    for key, val in summary_dict.items():
        tf.summary.scalar(key, val)
    summaries = tf.summary.merge_all()
    summary_dir = "logs/2D/{}/{}".format(args.dataset, args.model_name)
    train_summary_writer = tf.summary.FileWriter("{}/train".format(summary_dir))

    # Save args
    with open("{}/args.json".format(summary_dir), "w") as file:
        json.dump(vars(args), file)

    # Saver
    saver = tf.train.Saver(max_to_keep=1)
    checkpoint_dir = "checkpoints/2D/{}/{}".format(args.dataset, args.model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create grid of test points for plotting density
    plot_grid_res = 128
    if args.dataset == 'einstein':
        lims = np.array([0, 1, 0, 1])
    else:
        lims = 1.1 * np.array(
            [
                data.min(axis=0)[0],
                data.max(axis=0)[0],
                data.min(axis=0)[1],
                data.max(axis=0)[1],
            ]
        )
    xi, yi = np.mgrid[
        lims[0] : lims[1] : plot_grid_res * 1j, lims[2] : lims[3] : plot_grid_res * 1j
    ]
    test_points = np.vstack([xi.flatten(), yi.flatten()]).T

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n = 0
        while True:
            if n % args.summary_interval == 0:

                # Print summaries
                q_log_prob_curr, log_prob_est_curr, alpha_curr = sess.run(
                    (mean_proposal_log_prob_data, mean_log_prob_est_data, alpha)
                )
                print("Step {}".format(n))
                print("Log prob density model (est): {:.4f}".format(log_prob_est_curr))
                print("Log prob proposal model: {:.4f}".format(q_log_prob_curr))
                print("alpha: {:.4f}".format(alpha_curr))
                print()

                if args.save_summaries:
                    # Train summaries (single batch)
                    train_summaries_curr = sess.run(summaries)
                    train_summary_writer.add_summary(train_summaries_curr, n)

                if args.save_plots:

                    # Plots
                    log_density_curr, proposal_log_density_curr = sess.run(
                        (aem.log_prob_est_data, aem.proposal_log_prob_data),
                        {x_batch: test_points},
                    )
                    _, axarr = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

                    # Plot data histogram
                    ax = axarr[0]
                    hist_range = np.array([[lims[0], lims[1]], [lims[2], lims[3]]])
                    ax.hist2d(
                        data[:, 0],
                        data[:, 1],
                        normed=True,
                        bins=plot_grid_res,
                        range=hist_range,
                    )
                    ax.set_title("Data")

                    # Plot proposal density
                    ax = axarr[1]
                    ax.pcolormesh(
                        xi, yi, np.exp(proposal_log_density_curr).reshape(xi.shape)
                    )
                    ax.set_title("Proposal")

                    # Plot AEM density (estimated)
                    ax = axarr[2]
                    ax.pcolormesh(xi, yi, np.exp(log_density_curr).reshape(xi.shape))

                    # Enforce axis limits and remove ticks / ticklabels
                    for ax in axarr:
                        ax.set_xlim(lims[:2])
                        ax.set_ylim(lims[2:])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    ax.set_title("AEM")

                    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85)

                    # Save figure
                    plt.savefig("{}/density.png".format(summary_dir), dpi=300)
                    plt.close()

                if args.save_checkpoints:
                    saver.save(
                        sess,
                        "{}/model.ckpt".format(checkpoint_dir),
                        global_step=n,
                        write_meta_graph=False,
                    )

            # Do optimization op
            _, n = sess.run((optim_op, global_step))

            # Finish training
            if n == args.max_steps:
                print("Finished training noise model")
                break


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
