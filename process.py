import glob
import json
import os
from pathlib import Path

import SimpleITK as sitk
import torch

from predict import PredictModel

class DatacentricBaseline:
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # Path configurations
        self.input_path = os.getenv("INPUT_PATH", "/input/")
        self.output_path = os.getenv("OUTPUT_PATH", "/output/images/automated-petct-lesion-segmentation/")
        self.output_path_category = os.getenv("DATA_CENTRIC_OUTPUT_PATH", "/output/data-centric-model.json")
        self.nii_path = os.getenv("nnUNet_raw", "/nnUNet_raw/") + "Dataset131_Autopet3/imagesVa/"
        self.result_path = os.getenv("nnUNet_results", "/nnUNet_results/")
        self.weights_path = "/opt/algorithm/weights/"

        # Initialize model and paths
        self.ckpt_paths = glob.glob(os.path.join(self.weights_path, "*.ckpt"))
        self.tta = True
        self.sw_batch_size = 12
        self.random_flips = 1
        self.dynamic_tta = True
        self.max_tta_time = 220

        self.inferer = PredictModel(
            model_paths=self.ckpt_paths,
            sw_batch_size=self.sw_batch_size,
            tta=self.tta,
            random_flips=self.random_flips,
            dynamic_tta=self.dynamic_tta,
            max_tta_time=self.max_tta_time,
        )

        # Create directories if they do not exist
        Path(self.input_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.nii_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_path).mkdir(parents=True, exist_ok=True)

    def save_datacentric(self, value: bool):
        """
        Save the data-centric model info as a JSON file.
        """
        print(f"Saving datacentric json to {self.output_path_category}")
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = sitk.ReadImage(nii_input_path)
        sitk.WriteImage(img, mha_out_path, True)

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

    def load_inputs(self):
        """
        Read input images from /input/ and convert them.
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "SUV.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "CTres.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Write the prediction output to /output/ directory.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.output_path, "PRED.nii.gz"),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print(f"Output written to: {os.path.join(self.output_path, uuid + '.mha')}")

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
        Load inputs, perform prediction, and write the outputs.
        """
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.save_datacentric(True)
        self.write_outputs(uuid)

if __name__ == "__main__":
    DatacentricBaseline().process()
