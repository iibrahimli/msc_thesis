import argparse

import omegaconf

from arithmetic_lm.tokenizer import TOKENIZERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        default="arithmetic_lm/conf/config.yml",
        help="Path to the config file",
    )
    args, remaining_args = parser.parse_known_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    cli_args = omegaconf.OmegaConf.from_cli(remaining_args)
    cfg = omegaconf.OmegaConf.merge(cfg, cli_args)
    print(omegaconf.OmegaConf.to_yaml(cfg))
    tokenizer = TOKENIZERS[cfg.tokenizer.name](**cfg.tokenizer.get("args"))


if __name__ == "__main__":
    main()
