##### Pyton Environment Setup
If you are willing to use conda, the following steps should get you up and running with a workable environment for the mlwpy code.

1. Download and install anaconda.  https://www.anaconda.com/distribution/#download-section
2.  Install the following packages.  I did them one at a time to keep the dependency solver from having fits.

```bash
conda install pydotplus
conda install memory_profiler
conda install py-xgboost
conda install tensorflow
conda install -c conda-forge pymc3
conda install seaborn
conda install scikit-image
conda install opencv
```

3.  You may wish to setup specific environments for your book related code.  If so, keep reading.

##### Basic debugging tools for jupyter kernels:
Creating conda environments and hooking then to jupyter notebook kernels can be a bit tricky.  If you are trying to find a kernel in a notebook and you can't, you'll want to check these two diagnostic outputs:

```
% conda env list
% jupyter kernelspec list
```

If you need to know more about this, check out the docs.  WARNING:  do not blindly copy code from these links.  In some cases, they are using different versions of python (python 2) and/or different installation systems (pip).  If you start mixing and matching these, your head and your computer might explode:
  * [For jupyter kernels](https://ipython.readthedocs.io/en/latest/install/kernel_install.html)
  * [For conda envs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

##### Create some environments to run notebooks
```
# make a book_base environment and register kernel with jupyter
% conda create --name book_base_2020 \
     ipykernel keras memory_profiler notebook opencv \
     pydotplus py-xgboost scikit-learn scikit-image \
     seaborn    
% conda activate book_base_2020
% conda install -c conda-forge pymc3
% python -m ipykernel install --user --name book_base_2020
# you can now use book_base_2020 from inside your notebooks
```

PyMC3 started playing nicely with other packages, *but* when executing
the code examples in the final notebook, it was failing (some of the Theano
backend compilation was causing a segmentation fault).  It appears not to need
its own env at this point.  So, I've rolled everything into `book_base_2020`.

Now, in any environment, you should be able to do something like the following (which is in the `testem` script file).  The code simply executes all of the code in all of the notebooks.
```
BASE_NOTEBOOKS="02_Technical_Starter \
			         03_GettingStartedWithClassification \
			         04_GettingStartedWithRegression \
			         05_EvaluatingAndComparingLearners \
			         06_EvaluatingClassifiers \
			         07_EvaluatingRegressors \
			         08_MoreClassificationMethods \
			         09_MoreRegressionMethods \
			         10_Manual_Feature_Engineering \
			         11_Tuning_and_Pipelines \
			         12_Combining_Learners_Ensemble_Methods \
			         13_Feature_Engineering_II_Automated \
							 14_Feature_Engineering_III_Domain \
							 15_Connections_Between_Learners"

for curr_nb in $BASE_NOTEBOOKS; do
	jupyter nbconvert --to notebook \
	        --execute \
					--ExecutePreprocessor.kernel_name=book_base  \
					--ExecutePreprocessor.timeout=-1 \
					--output-dir=nbout/ \
					${curr_nb}_code.ipynb
done
```
