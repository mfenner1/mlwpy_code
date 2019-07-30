# mlwpy_code
Code from the Pearson Addison-Wesley book Machine Learning with Python for Everyone

### Pyton Environment Setup
If you are using conda, the following steps should get you up and running with a workable environment for the mlwpy code.

1. Download and install anaconda.  https://www.anaconda.com/distribution/#download-section
2. (Optional)  Downgrade python to version 3.6.  I'm not sure if I did this because of a specific issue or as a pre-cautionary step.  `conda install python=3.6`
3.  Install the following packages.  I did them one at a time to keep the dependency solver from having fits.  [See below for versioning information.]

```bash
conda install pydotplus
conda install memory_profiler
conda install py-xgboost
conda install tensorflow
conda install pymc3
conda install seaborn
conda install scikit-image
conda install opencv
```

### Package Versions

During the late development of the book, these are the versions of software that I fixed in place to keep my aim on a stable target.  You can install specific versions of packages using anaconda/conda [as documented here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages).  For example, `conda install scikit-learn=0.20`.  

My plan is to update the book software to the most recent versions of software around once a year (likely near August/September).  I'll update the software versions listed here when I do so.

```  
matplotlib                2.2.3            py36h54f8f79_0  
pandas                    0.23.4           py36h6440ff4_0
patsy                     0.5.1                    py36_0
pymc3                     3.5                      py36_0  
py-xgboost                0.80             py36h0a44026_0  
scikit-learn              0.20.0           py36h4f467ca_1  
seaborn                   0.9.0                    py36_0
statsmodels               0.9.0            py36h1d22016_0  
tensorflow                1.12.0       mkl_py36h2b2bbaf_0
```
To make a conda environment for the book code, I did the following:

```
conda create --name book
conda activate book
conda install jupyter
conda install scipy=1.2
conda install statsmodels=0.9
conda install scikit-learn=0.20
conda install seaborn=0.9
# <etc>
```

You may have to track down current documentation on adding an environment to your list of jupyter kernels.  That process has gone through a few different iterations lately, so it is unfortunately difficult for me to make general recommendations here.  On Mac, I edited my `Library/Jupyter/kernels` directory by hand (which I can't generally recommend because you might fry your Python installation and then hate me).
