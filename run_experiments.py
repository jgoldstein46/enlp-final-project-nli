from Trainer import NLI_Trainer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=50) # batch size before weights update
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    args = parser.parse_args()
    # convert to dict
    params = vars(args)
    trainer = NLI_Trainer(params)
    trainer.run_training_loop()
    