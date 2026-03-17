This project trains a neural network on custom data generated in Flightstream.

Data is stored in in the data directory, the cases subdirectory.
Resulting models are saved in the model directory.

## To run the program: ##
### Run dataPreprocessor.py ###
This takes the raw data from flightstream and puts it onto a c-grid.

### Run dataAugmenter.py ###
This takes all the data on c-grids and reflects it across the x-axis in order to create a larger data set.

### Run augmentedDataTrainer.py ###
This trains a model on the data that we made using dataPreprocessor.py and dataAugmenter.py

## Other useful programs: ##
### checkInputs.py ###
This just makes sure that the data is fine, it will be removed in future commits.

### dataResolutionCheck.py ###
This makes sure that the data provided from flighstream has a big enough resolution to have the data in the c-grid be mathematically sound.

### diagnosticaoa.py ###
This was used to check that the data worked, it will be removed in future commits.

### aggresiveTrainer.py ###
This was used as an earlier version of the training model, it will be removed in figure commits.

### generatedDataVisualizer.py ###
This shows you what the generated data looks like (note, does not show a prediction, shows ground truth to make sure that everything is working.)

### modelPredictionVisualizer.py ###
This shows an image of a model prediciton vs ground truth

### modelTrain.py ###
Old model training program, it will be removed in future commits
