# Dataset

This project uses the NIH ChestX-ray14 dataset.

## Download
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

Download the following:
- All `images_00x.tar.gz` files (12 parts)
- `Data_Entry_2017.csv` (contains image labels)

## How labels work
This dataset does not have Normal/Pneumonia folders.
Labels are inside the CSV file:
- `No Finding` → Normal
- `Pneumonia` → Pneumonia

## After downloading, organize like this:
data/
  images/
    00000001_000.png
    00000002_000.png
    ...
  Data_Entry_2017.csv