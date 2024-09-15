This code is aimed to Get Tumor Segmentation for AutoPetIII. The whole files can be download  using the Google Drive link provided：
https://drive.google.com/drive/folders/1IP8I2wduqSJrMcxZoJKBRgaZQs8zd0OY

Here, The nnUnet_raw has the test example file. If you want to test other test dataset of AutopetIII, you will need to add them to this file(autopet-iii-main\nnUnet_raw\Dataset131_Autopet3\imagesVa)
nnUNet_results has the trained model.
 The Docker code has been debug and the Docker construct process can  be seen as follow
 (1)cd \autopet-iii-main
 (2) docker build -t autopet-iii .
 (3) docker run --rm --gpus all autopet-iii
 
