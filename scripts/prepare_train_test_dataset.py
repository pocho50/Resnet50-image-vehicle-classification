"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """

    if not os.path.isdir(output_data_folder):
        os.mkdir(output_data_folder)

    with open(labels, "r") as lab_f:
        rowRount = 0
        csvreader = csv.reader(lab_f, delimiter=",")
        for row in csvreader:
            rowRount += 1
            # we avoid the first row
            if rowRount == 1:
                continue

            sub_folder = row[2]
            if not os.path.isdir(os.path.join(output_data_folder, sub_folder)):
                os.mkdir(os.path.join(output_data_folder, sub_folder))

            label_folder = row[1]
            label_folder_path = os.path.join(
                output_data_folder, sub_folder, label_folder
            )
            if not os.path.isdir(label_folder_path):
                os.mkdir(label_folder_path)

            image_name = row[0]
            image_link_path = os.path.join(
                output_data_folder, sub_folder, label_folder, image_name
            )
            image_path = os.path.join(data_folder, image_name)
            if not os.path.exists(image_link_path):
                os.link(image_path, image_link_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
