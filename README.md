# mlwpy_code
Code from the Pearson Addison-Wesley book Machine Learning with Python for Everyone

### Versioning Note
The code here is the latest/greatest version.  It has been updated to work with the most recent versions of its dependencies (e.g., scikit-learn).  If you want the version of the code that came with the book, you'll probably want [the 1.0 release](https://github.com/mfenner1/mlwpy_code/tree/v1.0).  Other releases are planned to occur annually around August.

### Python Environment Setup
See `make_environments.md` for details.

### Package Versions
Here are the versions of important packages as of the August 2019 code update.  As a reminder, you can install specific versions of packages using anaconda/conda [as documented here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages).  For example, `conda install scikit-learn=0.20`.  

My plan is to update the book software to the most recent versions of software around once a year (likely near August/September).  I'll update the software versions listed here when I do so.  The full list of package versions is available in `book_base_env.yml` produced by `conda env export > book_base_env.yml`.

```  
matplotlib                3.1.0  
numpy                     1.16.4
pandas                    0.25.0
patsy                     0.5.1
pymc3                     3.7     (conda-forge)  
py-xgboost                0.9  
scikit-learn              0.21.2  
scipy                     1.3.1
seaborn                   0.9.0
statsmodels               0.10.1  
tensorflow                1.14.0
```
