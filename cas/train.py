import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
# from unet3d.model.Ablation.EDRB_UNet import unet_model_3d
from unet3d.model.unet import unet_model_3d
# from unet3d.model.Segnet import unet_model_3d
# from unet3d.model.DenseVoxNet import unet_model_3d
# from unet3d.model.PE_bn_unet import unet_model_3d
# from unet3d.model.FullResUnet import unet_model_3d
# from unet3d.model.ResUnet import unet_model_3d
# from unet3d.model.EDDenseUnet import unet_model_3d
# from unet3d.model.RsEencodeDenseDecodeU_net import unet_model_3d
# from unet3d.model.EDResUnet import unet_model_3d
# from unet3d.model.FullResUnet import unet_model_3d
from unet3d.training import load_old_model, train_model
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations

config["patch_shape"] = (64, 64, 64)  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["raw-data"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 3
config["validation_batch_size"] = 6
config["n_epochs"] = 80 # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("coronary_data.h5")
config["model_file"] = os.path.abspath("coronary_segmentation_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
config["overwrite"] = True # If True, will previous files. If False, will use previously written files.
config["image_shape"] = (512, 512, 200)  # This determines what shape the images will be cropped/resampled to.
def fetch_training_data_files():
    training_data_files = list()
    #for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "cas", "*")):
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "Data",
                                              "冠脉树数据集1.0版本", "CAS_tree_test", "*")): #CAS_tree_test train-病人
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            #print(modality)
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files

def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files = fetch_training_data_files()

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"])#,
                              #batch_normalization=True)

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])