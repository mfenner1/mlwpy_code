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
Here are the versions of important packages as of the Feb 2024 code update.  As a reminder, you can install specific versions of packages using anaconda/conda [as documented here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages).  For example, `conda install scikit-learn=1.3.0`.  

My plan is to update the book software to the most recent versions of
software around once a year (likely near August/September).  Life's
been crazy, so the timeline has slipped a bit.  Sorry.  

I'll update the software versions listed here when I do so.  The full
list of package versions is available in `book_env_2024_feb.yml`
produced by `conda env export --from-history >
book_env_2024_feb.yml`.  Actually, it's not anymore because locking
package versions limits platform independence.  But, the important
ones are listed here.

```  
matplotlib                3.8.0
numpy                     1.22.3
opencv                    4.6.0
pandas                    2.0.3
patsy                     0.5.3
pymc3                     3.11.4
py-xgboost                1.7.3 
scikit-learn              1.3.0  
scipy                     1.7.3
seaborn                   0.12.2
statsmodels               0.14.0  
tensorflow                2.10.0
```
