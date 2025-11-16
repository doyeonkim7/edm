import json
import matplotlib.pyplot as plt
import traceback

def plot_loss_from_jsonl(jsonl_filepath, loss_key, image_filepath):
    """
    Reads loss data from a JSON Lines (.jsonl) file, plots the data, 
    and saves the plot to an image file.

    Args:
        jsonl_filepath (str): The path to the input JSON Lines file.
        loss_key (str): The key in the JSON object on each line that 
                        holds the loss value for that step/epoch.
        image_filepath (str): The path to save the output image file (e.g., 'loss_plot.png').
    """
    loss_values = []
    adjusted_loss_values = []
    kimg_values = []
    
    try:
        # 1. Load data from the JSON Lines file line by line
        with open(jsonl_filepath, 'r') as f:
            for line_number, line in enumerate(f, 1):
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 2. Extract the loss value
                    if 'Loss/loss' in data:
                        loss_values.append(data[loss_key]['mean'])
                    
                    if 'Loss/adjusted' in data:
                        adjusted_loss_values.append(data['Loss/adjusted']['mean'])
                    
                    if 'Progress/kimg' in data:
                        kimg_values.append(data['Progress/kimg']['mean'])
                            
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON on line {line_number} in {jsonl_filepath}. Skipping line.")
                    continue
        
        if not loss_values:
            print(f"Error: No valid loss values found with key '{loss_key}' in {jsonl_filepath}.")
            return

        # 3. Create the plot        
        plt.figure(figsize=(10, 6))
        plt.plot(kimg_values, loss_values, marker='.', linestyle='-', color='r', label='Loss')

        if adjusted_loss_values:
            plt.plot(kimg_values, adjusted_loss_values, marker='.', linestyle='-', color='b', label='Adjusted Loss')
        
        # Add titles and labels
        plt.title('Training Loss Progression', fontsize=16)
        plt.xlabel('kimg', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        
        plt.grid(True)
        plt.legend()
        
        # 4. Save the plot
        plt.savefig(image_filepath)
        plt.close()

        print(f"\n✅ Successfully plotted loss ({len(loss_values)} points) and saved the image to: **{image_filepath}**")

    except FileNotFoundError:
        print(f"Error: The file **{jsonl_filepath}** was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


balanced_path = 'experiments/balanced/00001-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'
adjusted_path = 'experiments/adjusted_1108/00002-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'
adjusted_lr_path = 'experiments/adjusted_1108/00003-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'
baseline_path = 'baseline_1111/00001-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'
adjusted_param_path = 'adjusted_param_1110/00001-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'
adjusted_1em2_path = 'adjusted_param_1e-2/00000-cifar10-32x32-uncond-ddpmpp-edm-gpus2-batch320-fp32/stats.jsonl'

plot_loss_from_jsonl(
    jsonl_filepath=adjusted_1em2_path,
    loss_key='Loss/loss', # This key must match the key in your .jsonl file
    image_filepath='adjusted_1em2_param.png'
)