# Environmental-Sound-Classification-ESC-using-neural-networks-and-other-classifiers

- Audio feature extraction and classification with the [ECS-10 data set](https://github.com/karoldvl/ESC-50) audio dataset 
- ECS-10 audio data is included. It consists of 10 classes of different environmental sounds (sea waves, kids playing, etc.)
- The main goal is to compare classification accuracies for the 6 tested classifiers. 

## Dependencies
- Librosa (audio loading, audio visualization and feature extraction)
- Sci-kit learn
- Keras (Theano backend)
- Numpy, Matplotlib
- Pandas (data visualization)

## Google Colab Notebook
A Google Colab Notebook (Python 3.7 Kernel) is added to illustrate the workflow. 

The scripts for feature extraction and classification have been added as 
```.ipynb``` files and are all loaded in the Jupyter Notebook sequentally.

Running ```feature_extraction.py``` creates a numpy array for features (```feature.npy```) and one for labels (```label.npy```).
These files will be saved in the current directory.

### Audio features extracted
- MFCC
- Chroma
- Mel spectrogram
- Tonal centroid feature
- Spectral contrast

### Classifiers implemented
- Convolutional Neural Network (CNN)
- Multilayer Perceptron (MLP)
- Recurrent Neural Network (RNN)
- Support Vector Machine (SVM)
- Random Forest (RF)
- Naive Bayes (NB)
- KNearestNeighbors(KNN)


### Accuracies obtained
**Note**: Direct comparison between classifiers can't be done yet since their parameters haven't been tuned to optimize
accuracy yet. Out of 400 audio samples, the test set consisted on the 33% of this.
- CNN: 78.125% (100 epochs)
- MLP: 79.125 (100 epochs)
- RNN: 72% (100 epochs)
- SVM: 81.7%
- RF: 83%
- NB: 69.7%
- KNN: 67%
### Approaches to improve accuracy
- Compute other features: MFCC + ZCR features improve classification 
[accuracy](https://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_3.pdf)
for speech, noise and music labels. See if it also works for the 10 classes.
- Tune optimization hyperparameters (for every classifier): Weight initialization, decaying learning rate.
- Data scaling and feature normalization (MFCC)
