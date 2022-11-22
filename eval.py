import argparse
import os
import numpy as np

from src.handlers.evaluater import Evaluator


if __name__ == '__main__':
    ### Decoding arguments
    eval_parser = argparse.ArgumentParser(description='Arguments for training the system')
    eval_parser.add_argument('--path', type=str, help='path to experiment')
    eval_parser.add_argument('--dataset', default='race++', type=str, help='dataset to train the system on')
    eval_parser.add_argument('--mode', default='test', type=str, help='which data split to evaluate on')
    eval_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')
    eval_parser.add_argument('--ensemble', action='store_true', help='whether ensemble evaluation')
    args = eval_parser.parse_args()

    print(args)
    # ensemble evaluation
    if args.ensemble:
        accs = []
        for seed in os.listdir(args.path):
            # Get seed Path
            seed_path = os.path.join(args.path, seed)

            # Generate predictions
            evaluator = Evaluator(seed_path, args.device)
            preds = evaluator.load_preds(args.dataset, 'test')

            # Run Evluation
            labels = evaluator.load_labels(args.dataset, 'test')
            acc = evaluator.calc_acc(preds, labels)
            accs.append(acc)

            #print seed performance 
            print(seed, acc)

        # Print performance
        mean, std = np.mean(accs), np.std(accs)
        print(f'ACC  : {mean:.1f} +- {std:.1f}')

    # single seed evaluation
    else:
        evaluator = Evaluator(args.path, args.device)
        preds = evaluator.load_preds(args.dataset, args.mode)
        labels = evaluator.load_labels(args.dataset, args.mode)
        acc = evaluator.calc_acc(preds, labels)
        print(f'accuracy: {acc:.2f}')
