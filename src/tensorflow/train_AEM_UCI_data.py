"""Train AEM on UCI + BSDS300 datasets."""
import tensorflow as tf
import numpy as np
import argparse
import json
import os

from utils.data_utils import UCI
from utils.aem import AEM


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
    parser.add_argument("--save_summaries", type=int, default=1, help="Save summaries")
    parser.add_argument(
        "--save_checkpoints", type=int, default=1, help="Save checkpoints"
    )
    parser.add_argument(
        "--summary_interval", type=int, default=2500, help="Summary save interval"
    )
    # ResMADE options
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
        default=512,
        help="Number of hidden units for res blocks in MADE",
    )
    parser.add_argument(
        "--dropout_p_MADE", type=float, default=0.1, help="Dropout probability in MADE"
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
        default=0.1,
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
        "--train_AEM",
        type=int,
        default=1,
        help="Train the AEM. If 0 then proposal model only is trained ",
    )
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
        "--batch_size", type=int, default=256, help="Batch size during training"
    )
    parser.add_argument(
        "--batch_size_val",
        type=int,
        default=512,
        help="Batch size for validation set eval",
    )
    parser.add_argument(
        "--use_subset_val",
        type=int,
        default=0,
        help="Use smaller val set for faster training",
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
    x_batch, data_val, _ = UCI(args.dataset, args.batch_size)
    if args.use_subset_val:
        data_val = data_val[:2500]

    # Create dropout placeholders
    dropout_p_MADE = tf.placeholder_with_default(0.0, ())
    dropout_p_energy_net = tf.placeholder_with_default(0.0, ())

    # Build AEM
    aem = AEM(
        x_batch,
        n_importance_samples=args.n_importance_samples,
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

    # Get log prob of data under model and proposal
    mean_log_prob_est_data = tf.reduce_mean(aem.log_prob_est_data)
    mean_proposal_log_prob_data = tf.reduce_mean(aem.proposal_log_prob_data)

    # Loss is alpha-weighted sum of log prob of data under energy model and proposal
    global_step = tf.Variable(0, trainable=False)
    if args.alpha_warm_up_steps > 0:
        alpha = tf.cast(global_step > args.alpha_warm_up_steps, tf.float32)
    else:
        alpha = tf.constant(1.0)
    if args.train_AEM:
        loss = -(alpha * mean_log_prob_est_data + mean_proposal_log_prob_data)
    else:
        loss = -mean_proposal_log_prob_data

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
    summary_vals = tuple([val for val in summary_dict.values()])
    summary_dir = "logs/UCI/{}/{}".format(args.dataset, args.model_name)
    train_summary_writer = tf.summary.FileWriter("{}/train".format(summary_dir))
    val_summary_writer = tf.summary.FileWriter("{}/val".format(summary_dir))

    # Save args
    with open("{}/args.json".format(summary_dir), "w") as file:
        json.dump(vars(args), file)

    # Saver
    saver = tf.train.Saver(max_to_keep=1)
    checkpoint_dir = "checkpoints/UCI/{}/{}".format(args.dataset, args.model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n = 0
        val_log_prob_best = -np.inf
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

                    # Val summaries
                    val_summaries_np = np.zeros(len(summary_vals))
                    n_summary_ex = data_val.shape[0]
                    n_val_batches = np.ceil(n_summary_ex / args.batch_size_val).astype(
                        "int"
                    )
                    for b in range(n_val_batches):
                        val_batch = data_val[
                            b * args.batch_size_val : (b + 1) * args.batch_size_val
                        ]
                        val_batch_size_curr = val_batch.shape[0]
                        val_summaries_np += val_batch_size_curr * np.array(
                            sess.run(summary_vals, {x_batch: val_batch})
                        )
                    val_summaries_np /= n_summary_ex
                    val_summary_dict = {
                        key: val for key, val in zip(summary_vals, val_summaries_np)
                    }
                    val_summaries_curr = sess.run(summaries, val_summary_dict)
                    val_summary_writer.add_summary(val_summaries_curr, n)

                    if args.save_checkpoints:
                        val_log_prob_curr = val_summary_dict[mean_log_prob_est_data]
                        if val_log_prob_curr > val_log_prob_best:
                            saver.save(
                                sess,
                                "{}/best_val_model.ckpt".format(checkpoint_dir),
                                global_step=n,
                                write_meta_graph=False,
                            )
                            val_log_prob_best = val_log_prob_curr

            # Do optimization op
            _, n = sess.run(
                (optim_op, global_step),
                {
                    dropout_p_MADE: args.dropout_p_MADE,
                    dropout_p_energy_net: args.dropout_p_energy_net,
                },
            )

            # Finish training
            if n == args.max_steps:
                print("Finished training noise model")
                break


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

