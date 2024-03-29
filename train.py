#! /usr/bin/env python
"""
Main training file which needs to be run.
"""
import numpy as np
import torch
from torch.autograd import profiler
import tqdm

from radioNN.networks.antenna_skipfc_network import AntennaNetworkSkipFC
from radioNN.process_network import NetworkProcess
from radioNN.tests.draw_graph import draw_graph


def main(percentage=0.1, base_path="./runs/"):
    """
    Run Code.
    Returns
    -------

    """
    process = NetworkProcess(
        model_class=AntennaNetworkSkipFC,
        # one_shower=one_shower,
        percentage=percentage,
        batch_size=8,
        base_path=base_path,
    )
    num_epochs = 500
    process.full_training(num_epochs)


def profile():
    """
    Profile code.
    Returns
    -------

    """

    process = NetworkProcess()
    _ = process.train()  # warmup
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        _ = process.train()
        print("ONE EPOCH DONE")

    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=5
        )
    )


def one_shower_training(one_shower=1):
    """
    Run Code for just one shower to see if memorizes the shower.
    Returns
    -------

    """
    print(f"Use shower {one_shower}")
    process = NetworkProcess(one_shower=one_shower)
    num_epochs = 1000

    for epoch in tqdm.autonotebook.trange(num_epochs):
        train_loss = process.train()
        tqdm.tqdm.write(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {train_loss}")
        process.scheduler.step(train_loss)

    torch.save(process.model.state_dict(), "antenna_network.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--percentage",
        default=0.1,
        type=float,
        help="Percentage of dataset to use",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default="./runs/",
        type=str,
        help="Base path for storing model information",
    )
    parser.add_argument(
        "--one_shower",
        action="store_true",
        help="Try to " "memorize a single shower",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the network for performance hotspots",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Graph the network for bad gradients",
    )
    opt = parser.parse_args()
    if opt.one_shower:
        print("Training with a single shower")
        one_shower_training(np.random.randint(low=1, high=2159))
        exit()
    if opt.profile:
        print("Profiling Code")
        profile()
        exit()
    if opt.graph:
        print("Drawing the computation graph")
        draw_graph()
        exit()
    print("No options provided, executing main training")
    main(percentage=opt.percentage, base_path=opt.base_path)
