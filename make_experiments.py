import argparse
import yaml
import ast


def parse_args():
    """
    Parse command-line arguments to update DATASET parameters.
    """
    parser = argparse.ArgumentParser(description="Update DATASET parameters in configuration")

    # Define arguments for DATASET parameters
    parser.add_argument("--num-keypoints", type=int, help="Set number of keypoints")
    parser.add_argument("--flip-fairs", type=str, help="Set flip fairs option as a string representation of a list")
    parser.add_argument("--data-format", type=str, help="Set data format")
    parser.add_argument("--data-root", type=str, help="Set data root")
    parser.add_argument("--begin-epoch", type=int, help="Set begin epoch")
    parser.add_argument("--end-epoch", type=int, help="Set end epoch")
    parser.add_argument("--output-file", type=str, default="updated_config.yaml", help="Path to save updated config")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the original configuration file") 

    return parser.parse_args()


def load_config(file_path):
    """
    Load YAML configuration from a file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def save_config(config, file_path):
    """
    Save YAML configuration to a file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Configuration saved to {file_path}")


def update_dataset_params(config, args):
    """
    Update DATASET parameters in the configuration based on command-line arguments.
    """
    if 'DATASET' not in config:
        config['DATASET'] = {}

    # Update parameters if arguments are provided

    if args.num_keypoints is not None:
        config['DATASET']['NUM_KEYPOINTS'] = args.num_keypoints

    if args.flip_fairs is not None:
        # Convert the string representation of the list to an actual list
        config['DATASET']['FLIP_FAIRS'] = ast.literal_eval(args.flip_fairs)

    if args.data_format is not None:
        config['DATASET']['DATA_FORMAT'] = args.data_format

    if args.data_root is not None:
        config['DATASET']['ROOT'] = args.data_root

    if args.begin_epoch is not None:
        config['TRAIN']['BEGIN_EPOCH'] = args.begin_epoch

    if args.end_epoch is not None:
        config['TRAIN']['END_EPOCH'] = args.end_epoch

    print("Updated DATASET parameters:")
    for key, value in config['DATASET'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Load the original configuration
    config = load_config(args.config_file)

    # Update DATASET parameters
    update_dataset_params(config, args)

    # Save the updated configuration to a file
    save_config(config, args.output_file)
