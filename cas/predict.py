import os
from train import config
import tensorflow as tf
# from unet3d.prediction_multi_loss import run_validation_cases
# from unet3d.prediction_multi_loss import run_validation_cases    #DenseVoxNet
from unet3d.prediction import run_validation_cases    #others


def main():
    prediction_dir = os.path.abspath("hybridlossffr/")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,    #if False, get probability map
                         threshold=0.5,
                         output_dir=prediction_dir)

    run_validation_cases(validation_keys_file=config["training_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,  # if False, get probability map
                         threshold=0.5,
                         output_dir=prediction_dir)

    # prediction_dir = os.path.abspath("FFR_DL_prediction_p/train/CHEN XIAO DONG")
    # run_validation_cases(validation_keys_file=config["validation_file"],
    #                      model_file=config["model_file"],
    #                      training_modalities=config["training_modalities"],
    #                      labels=config["labels"],
    #                      hdf5_file=config["data_file"],
    #                      output_label_map=False,  # if False, get probability map
    #                      output_dir=prediction_dir)


if __name__ == "__main__":
    main()
