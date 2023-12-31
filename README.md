# Transients-MissingData
-----------------------------------------------------------------
## Can a Deep Neural Network Classify Astrophysical Transients? 
------------------------------------------------------------------

Revisiting the Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC). [Challenge Paper](https://arxiv.org/abs/1810.00001)

* How to deal with missing values in time series?
* Can a Transformer-based network reach a reasonable classification score?
* How does the performance compare with some Seq2Seq or 1D convolution? 

-------------------------------------------------------------------

## Notes

* The script **Interpolate_and_Create_Dataset_GaussianProcess.py** is the one that will generate the *.h5* file needed to train the models. This script will take as input the different CSV needed from the *Data/* folder. As it is written right now, it generates all the data (galactic + extragalactic). I will separate them into 3 different scripts if needed (one for the total data, one for galactic and another for extragalactic only).
* The script **Interpolate_and_Create_Dataset_ZeroFilling.py** does more or less the same except for the fact that the interpolation is not the Gaussian process but the "vanilla" one where we fill with zeroes.
* The script **Obtain_Fourier_transform_from_Timeseries.py** will read the *.h5* generated by the other scripts and output only the data corresponding to the Fast Fourier Transform.

All the scripts generate different datasets within the *.h5* file, where the separation into training, validation and test datasets is already considered.

* The script **models.py** contains the different classes from PyTorch defining the models, and **utils.py** contains miscellaneous stuff such as plotting functions, etc.
* The tests considered are contained in the *Different_Tests/* folder, with a brief description of what has been done in the TXT file.
