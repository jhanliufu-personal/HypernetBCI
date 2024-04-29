from utils import parse_training_config

args = parse_training_config()
print(args.model_name)
print(args.trial_len_sec)
print(args.significance_level)