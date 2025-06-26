from Casual_SAC import SACAgent
import pandas as pd
from env import ADEnv
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    task_name = args.task_name
    Data_Path = 'datasets/'
    training_set = pd.read_csv(data_path).values
    testing_set = pd.read_csv(test_data_path).values

    env = ADEnv(
        dataset=training_set,
        sampling_Du=args.sampling_du,
        prob_au=args.prob_au,
        label_normal=args.LABEL_NORMAL,
        label_anomaly=args.LABEL_ANOMALY,
        TC=args.TC,
        TH=args.TH,
        device=device,
        deque_len=args.deque_len,
        TH_update_interval=args.TH_update_interval
    )
    state_dim = env.n_feature
    action_dim = 1

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        env=env, device=device,
        test_dataset=testing_set,
        actor_lr=args.lr,
        critic_lr=args.lr,
        gamma=0.99,
        tau=args.tau,
        warmup_steps=args.warmup_steps,
        step_per_episode=args.step_per_episode,
        max_memory_size=args.max_memory_size,
        alpha_lr=0.0003,
        gat_dim=args.gat_dim,
        mlp_hidden_dim=args.mlp_dim
    )
    agent.fit()
    agent.show_results()
    roc, pr = agent.model_performance()
    print(f'Finished run with pr: {pr} and auc-roc: {roc}...')


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--task_name',type=str, default="ANN", help="Task name")
    parser.add_argument('--prob_au', type=float, default=0.5, help="Probability for 'au'")
    parser.add_argument('--sampling_du', type=int, default=1000, help="Sampling for 'Du'")
    parser.add_argument('--LABEL_NORMAL', type=int, default=0, help="Label for normal data")
    parser.add_argument('--LABEL_ANOMALY', type=int, default=1, help="Label for anomaly data")
    parser.add_argument('--TC', type=int, default=4, help="TC value for ADEnv")
    parser.add_argument('--TH', type=float, default=0.8, help="TH value for ADEnv")
    parser.add_argument('--deque_len', type=int, default=50, help="Deque length for ADEnv")
    parser.add_argument('--TH_update_interval', type=int, default=10, help="TH update interval for ADEnv")

    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate for actor")
    parser.add_argument('--critic_lr', type=float, default=0.0001, help="Learning rate for critic")
    parser.add_argument('--gamma', type=float, default=0.99, help="Gamma value for SACAgent")
    parser.add_argument('--tau', type=float, default=0.2, help="Tau value for SACAgent")
    parser.add_argument('--warmup_steps', type=int, default=10000, help="Warmup steps for SACAgent")
    parser.add_argument('--step_per_episode', type=int, default=5000, help="Steps per episode for SACAgent")
    parser.add_argument('--max_memory_size', type=int, default=10000, help="Maximum memory size for SACAgent")
    parser.add_argument('--alpha_lr', type=float, default=0.0003, help="Alpha learning rate for SACAgent")
    parser.add_argument('--gat_dim', type=int, default=256, help="GAT dimension for SACAgent")
    parser.add_argument('--mlp_dim', nargs='+', type=int, default=[32,16], help="GAT dimension for SACAgent")
    args = parser.parse_args()

    main(args)

