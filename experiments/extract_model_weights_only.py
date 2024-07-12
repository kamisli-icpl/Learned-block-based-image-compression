import os
import torch

def process_checkpoint_files(base_dir):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Check if 'checkpoints' directory exists in the current directory
        if 'checkpoints' in dirs:
            checkpoints_dir = os.path.join(root, 'checkpoints')
            
            # Iterate over files in the 'checkpoints' directory
            for file_name in os.listdir(checkpoints_dir):
                if file_name.endswith('pth.tar'):
                    file_path = os.path.join(checkpoints_dir, file_name)
                    
                    try:
                        # Load the checkpoint file
                        checkpoint = torch.load(file_path)
                        
                        # Check if 'state_dict0' key exists in the checkpoint
                        if 'state_dict0' in checkpoint:
                            state_dict = checkpoint['state_dict0']
                            state = {'state_dict0': state_dict}
                            
                            # Save the state_dict with the new file name
                            save_path = os.path.join(checkpoints_dir, 'model_best_acl_Xrr.pth.tar')
                            torch.save(state, save_path)
                            print(f"Saved state_dict0 to {save_path}")
                        else:
                            print(f"'state_dict0' key not found in {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    base_directory = '.'  # Current directory
    process_checkpoint_files(base_directory)

