# mlwpy_code
Code from the Pearson Addison-Wesley book Machine Learning with Python for Everyone

### Versioning Note
The code here is the latest/greatest version.  Updated releases are planned annually in August.  The version here has been updated to work with the most recent versions of its dependencies (e.g., scikit-learn and pandas). If you want a quick link to the latest release (with convenient downloads), [click here](https://github.com/mfenner1/mlwpy_code/releases/latest).

If you want the exact version of the code that came with the book, you'll want the 1.0 release which you can
  * [view the source of](https://github.com/mfenner1/mlwpy_code/tree/v1.0)
  * [or download directly](https://github.com/mfenner1/mlwpy_code/releases/tag/v1.0).  


### Python Environment Setup
See `make_environments.md` ([here](https://github.com/mfenner1/mlwpy_code/blob/master/make_environments.md)) for details.

### Package Versions
Here are the versions of important packages as of the Nov. 2020 code update.  As a reminder, you can install specific versions of packages using anaconda/conda [as documented here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages).  For example, `conda install scikit-learn=0.23.2`.  

My plan is to update the book software to the most recent versions of software around once a year (likely near August/September).  This year (2020), that slipped a bit.  Sorry.  

I'll update the software versions listed here when I do so.  The full list of package versions is available in `book_base_2020_env.yml` produced by `conda env export > book_base_2020_env.yml`.

```  
matplotlib                3.3.2
numpy                     1.19.2
opencv                    3.4.2
pandas                    1.1.3
patsy                     0.5.1
pymc3                     3.9.3     (conda-forge)  
py-xgboost                0.9  
scikit-learn              0.23.2  
scipy                     1.5.2
seaborn                   0.11.0
statsmodels               0.12.0  
tensorflow                1.14.0
```
