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
% conda create --name book_base \
     ipykernel keras memory_profiler notebook opencv \
     pydotplus py-xgboost scikit-learn scikit-image \
     seaborn    
% conda activate book_base
% python -m ipykernel install --user --name book_base
```

PyMC3 was not playing nicely with others, so I decided to have it
standalone in its own env.  
```
# make a book_pymc3 env and register kernel with jupyter
# using conda-forge channel b/c pymc3 breaks with main channel
# i don't install notebook here, but you could
% conda create --name book_pymc3 \
        -c conda-forge \
        arviz ipykernel mkl-service pymc3 scikit-learn seaborn
% conda activate book_pymc3
% python -m ipykernel install --user --name book_pymc3
```

Now, in any environment, you should be able to do something like the following (which is in the `testem` script file):
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

# pymc3 notebook
jupyter nbconvert --to notebook \
				--execute \
				--ExecutePreprocessor.kernel_name=book_pymc3  \
				--ExecutePreprocessor.timeout=-1 \
				--output-dir=nbout/ \
				15_Connections_Between_Learners_pymc3_code.ipynb
```
