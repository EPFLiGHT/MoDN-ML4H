# Modular Clinical Decision Support Networks (MoDN) 
This is the code accompanying the *Modular Clinical Decision Support Networks (MoDN) Updatable, Interpretable, and Portable Predictions for Evolving Clinical Environments* abstract.

## Plots
The experiments were run on python version 3.8.10. The **data** folder must contain the data (link to anonymized data https://zenodo.org/record/400380#.Yug5kuzP00Q) and the **models** folder contains the scripts used to run the different experiments and produce the plots.

The script **main.py** calls the preprocessing pipeline on the data, saves the preprocessed data (qst_obj) and trains the model either performing 5 times 2-fold CV (saving the different metrics) or just training a single model. 

The script **iio_training.py** performs the IIO experiments (i.e. compartmentalization and fine tuning). The different models and performance scores are saved to the **updated_centralized** folder.

After having run both these files, one can run **statistical_tests.py** to produce the plots (uses saved metrics by the two previous scripts). 

Metrics and performance scores are saved to the *saved_objects* folder and plots to the *saved_plots* folder.


## Other files
**baselines.py** contains the functions to compute the KNN and logistic regression baselines. 

**dataset_generation.py** puts the data in shape to be used by the models. 

**distributed_training_parameters.py** contains the parameters used by the **distributed_training.py** file. 

**graph_functions.py** contains many functions to produce some plots. 

**modules.py** contains the module and state definitions. 

**training_procedures_epoct.py** contains the training and testing processes for the model. 

**utils_distributed.py** contains some utlitary functions for the compartmentalisation and fine-tuning experiments. 

**utils_epoct.py** contins utilitary functions specific to the epoct data and **utils.py** general utilitary functions. 

## Reproducing the results
To reproduce the results reported in *Modular Clinical Decision Support Networks (MoDN) Updatable, Interpretable, and Portable Predictions for Evolving Clinical Environments*, install the necessary dependencies using:

`sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super`

`pip install -r requirements.txt` (from the root directory)

Then run the different scripts as described in the **Plots** paragraph.





