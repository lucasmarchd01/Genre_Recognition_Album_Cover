# Music Genre Classification using Album Covers

This project aims to classify music genres using album cover images. The classification is based on deep learning techniques, specifically convolutional neural networks (CNNs), trained on a dataset of album cover images associated with different music genres. The dataset is generated from the [The AcousticBrainz Genre Dataset](https://mtg.github.io/acousticbrainz-genre-dataset/).

## Overview

The project consists of three main components:

- Data Retrieval:
    - The retrieve_art.py script retrieves album cover images from the [Cover Art Archive API](https://wiki.musicbrainz.org/Cover_Art_Archive/API) based on MusicBrainz release group IDs (MBIDs) stored in a tab-separated values (TSV) file. 
    - It downloads the images, stores them locally, and saves the file paths to a CSV file.
    - The TSV file can be found [here](https://zenodo.org/records/2554044)

- Data Processing:  
    - The data_processing.py script preprocesses the data by mapping genre labels to MBIDs and adjusting image file paths. It also filters the data to only include the top 6 most frequent genre classes.

- Model Training:
    - The train.py script trains a CNN model on the preprocessed dataset. It uses the VGG16 pre-trained model as the base and fine-tunes it on the album cover images.

## Requirements

- Clone the repository:
```
git clone https://github.com/lucasmarchd01/Genre_Recognition_Album_Cover.git
cd Genre_Recognition_Album_Cover
```
- Create a conda environment:
`conda create -n "myenv" python=3.10` (or use your preffered virtual environment)

- Install the required dependencies:

`pip install -r requirements.txt`


## Usage

### Running locally (CPU)

Run the data retrieval and preprocessing scripts:

`python retrieve_art.py <tsv-filename>`

`python data_processing.py`

Train the model:

`python train.py <directory-path>`

Replace `<tsv-filename>` with the path to the TSV file containing MBIDs, and `<directory-path>` with the path to the directory containing the dataset (`./` in this case).

### Running on Digital Research Alliance of Canada (GPU)

- Archive the dataset (the `/data` directory):

`$ tar cf mydataset.tar data/*`
- Load modules required by TensorFlow:

`[name@server ~]$ module load python/3.10 cuda/12.2 cudnn/8.9.5.29`
- Create a new Python virtual environment:

`[name@server ~]$ virtualenv --no-download tensorflow`
- Activate Python virtual environment:

`[name@server ~]$ source tensorflow/bin/activate`
- Install TensorFlow:

`(tensorflow) [name@server ~]$ pip install --no-index tensorflow`
- Install requirements from `requirementsvenv.txt`:

`pip install -r requirementsvenv.txt --no-index`
- Submit a job using the supplied bash script and `sbatch` command (this example is on the Cedar cluster):

`sbatch genre-recognition.sh --gres=gpu:1 --cpus-per-task=6 --mem=32000M --time=6:00:00`

## Directory Structure
```
.
├── data
│   ├── csv
│   │   ├── final_top_6.csv
│   │   ├── mbid_to_image_filenames.csv
│   │   └── output_interrupted.csv
│   └── images
│       ├── <image-files>
├── logs
│   └── art_retrieval.log
├── results
│   ├── best_model.keras
│   ├── confusion_matrix.png
│   ├── results.txt
│   ├── training_validation_accuracy.png
│   └── training_validation_loss.png
├── retrieve_art.py
├── data_processing.py
├── train.py
├── genre-recognition.sh
└── README.md
```

## Results

The trained model is saved in the results directory along with evaluation metrics such as the confusion matrix and classification report.
- best_model.keras 
- confusion_matrix.png
- results.txt, which contains:
    - test loss, test accuracy, classification report (precision, recall, f1 score for each class) and confusion matrix
- training and validation accuracy over each epoch of training
- training and validation loss over each epoch of training

## Credits

This project utilizes data from the [Cover Art Archive](https://coverartarchive.org/) and the [MusicBrainz](https://musicbrainz.org/) database.

```
Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale. 
20th International Society for Music Information Retrieval Conference (ISMIR 2019).
```
