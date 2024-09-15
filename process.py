import os
from pathlib import Path
import torch
import shutil
import uuid
import SimpleITK as sitk

class Autopet:
    def __init__(self):
        """
        Initialize paths and parameters.
        """
        # Read environment variables or use defaults
        self.input_path = os.getenv("INPUT_PATH", "/input/")
        self.output_path = os.getenv("OUTPUT_PATH", "/output/images/automated-petct-lesion-segmentation/")
        self.nii_path = os.getenv("nnUNet_raw", "/nnUNet_raw/") + "Dataset131_Autopet3/imagesVa/"
        self.result_path = os.getenv("nnUNet_results", "/nnUNet_results/")  # nnUNet results path
        self.nii_seg_file = "Dataset131_Autopet3.nii.gz"

        # Print paths for debugging
        print("INPUT_PATH:", self.input_path)
        print("OUTPUT_PATH:", self.output_path)
        print("nnUNet_raw_data_base:", os.getenv("nnUNet_raw"))
        print("nnUNet_results:", os.getenv("nnUNet_results"))
        print("NII Path:", self.nii_path)
        print("Result Path:", self.result_path)

        # Create directories if they do not exist
        Path(self.input_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.nii_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_path).mkdir(parents=True, exist_ok=True)

    def check_gpu(self):
        """
        Check if GPU is available.
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        if is_available:
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print("Device memory: " + str(torch.cuda.get_device_properties(0).total_memory))

    def predict(self):
        """
        Perform prediction using nnUNet.
        """
        os.system(
            f"nnUNetv2_predict -i {self.nii_path} "
            f"-o {self.output_path} "
            f"-d 131 -c 3d_fullres_resenc_bs9 -f 0 -step_size 0.6 --save_probabilities"
        )
        print("Prediction finished")

    def process(self):
        """
        Process the input data, perform prediction, and write the results to /output/ directory.
        """
        self.check_gpu()
        print("Start processing")
        self.predict()

if __name__ == "__main__":
    Autopet().process()
