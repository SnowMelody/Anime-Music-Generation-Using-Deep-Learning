<h1><p align="center"> Anime Music Generation Using Deep Learning </p></h1>

This project uses bar counter, repeated motifs, and form as conditioning inputs, with a model architecture of Bi-LSTM and attention layers to generate anime music with better structure. 2 other models are also provided for comparison.

## Contents 
- [Getting started](#getting-started)</br>
  - [Prerequisites](#prerequisites)</br>
- [User guide](#user-guide)</br>
  - [Transposing the data](#transposing-the-data)</br>
  - [Form](#form)</br>
  - [Bar counter and repeated motifs](#bar-counter-and-repeated-motifs)</br>
  - [Running the proposed model](#running-the-proposed-model)</br>
  - [Running the other 2 models](#running-the-other-2-models)</br>
  - [Other stuff](#other-stuff)</br>

## Getting started

### Prerequisites
The code runs in Python 3 environment. Dependencies are listed in _dependencies.txt_. To install a specific dependency, type the following in command line: </br>

```
pip install dependency_name
```

Download the file, unzip it, and move the folder to your preferred location. Then, open up command line and ```cd``` to the location where you have placed the folder. For example: </br>

```
cd D:\username\Desktop\Anime-Music-Generation-Using-Deep-Learning
```

You're good to go. </br>


## User guide
The following sections provide a guide on how to run the program. This allows you to follow the pipeline of the program, or apply your own data on the program. </br>

### Transposing the data
The original MIDI files and sheet music are located in _midi_files_, and transposed files are located in _transposed_midi_files_. To transpose MIDI files, save your MIDI files in _midi_files_, run the script _transpose_midi.py_, and the transposed files will be saved in _transposed_midi_files_. </br>

### Form
The form for MIDI files are created based on their sheet music and stored in script _FORM_INPUT.py_. This script is read into the main scripts for various models. To use your own form, ensure that it follows the same format in the form script. That is, form notations are done in divisions of a **bar length**. </br>

### Bar counter and repeated motifs 
These other 2 conditioning inputs are automatically processed by the functions in the main script, which should not require any editing. The functions in the script are _get_bar_count_ and _get_motifs_. As bar counter processes in 4/4 time, if your MIDI files have other time signatures then you'll have to edit the function. Sliding window and maximum count parameters can be changed in _get_motifs_.

### Running the proposed model
To train the proposed model, run the script _model1 (proposed).py_. To generate music samples, run the script _model1_generate_ after the model has been trained. During every training epoch, the model's checkpoint and weights will be saved in _training_checkpoints_model1_. The model's loss and accuracy can be analysed and plotted using _plot_model_history.py_. </br>

### Running the other 2 models
Model 2 is the model without conditioning, and model 3 is the baseline LSTM model. Their scripts/folders are labelled and run the same way as the proposed model. Do take note that conditioning inputs were processed for these models but **not** used in order to obtain the same train-test splits across all 3 models. So if you are using your own data but without conditioning information, simply edit the script part for train-test split and comment out the portions used to process or read in conditioning information. </br>

### Other stuff
Sources used to obtain anime MIDI files and sheet music are listed in an excel file in folder _excel_file_anime_songs_used_. Detailed results of the user study is stored as an excel file in folder _excel_file_user_study_results_. </br>
