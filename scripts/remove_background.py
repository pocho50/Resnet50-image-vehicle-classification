"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os, cv2
from utils import utils as u, detection


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    #   2. Load the image
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.

    # create the output_data_folder folder
    os.makedirs(output_data_folder, exist_ok=True)

    for dirpath, filename in u.walkdir(data_folder):
        # get subset name
        subset_name = os.path.basename(os.path.dirname(dirpath))
        # get class name
        class_name = os.path.basename(os.path.dirname(os.path.join(dirpath, filename)))

        # create the subset and class folder
        os.makedirs(os.path.join(output_data_folder, subset_name), exist_ok=True)
        os.makedirs(
            os.path.join(output_data_folder, subset_name, class_name), exist_ok=True
        )

        if os.path.exists(
            os.path.join(output_data_folder, subset_name, class_name, filename)
        ):
            continue

        # create the image from array
        img_array = cv2.imread(os.path.join(dirpath, filename))

        # get the box coordinates
        box_coordinates = detection.get_vehicle_coordinates(img_array)

        # generate the new image from box coordinates
        new_img_array = img_array[
            box_coordinates[1] : box_coordinates[3],
            box_coordinates[0] : box_coordinates[2],
            :,
        ]

        cv2.imwrite(
            os.path.join(output_data_folder, subset_name, class_name, filename),
            new_img_array,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
