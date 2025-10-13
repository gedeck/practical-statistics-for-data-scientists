## Instructions and Errors

Tested using Python 3.8.5 on Mac M1 and 3.8.11 on Mac Intel processor

All:
```
$ pip install pandas
$ pip install sklearn
$ pip install dmba
$ pip install matplotlib
$ pip install seaborn
```
Then run the Python files from the command line, e.g.:

`$ ch_1_01_states_populations_and_murders.py`

At the top of the ch_*.py files is `#!/usr/local/bin/python`.
This should point to Python 3.

#### Chapter 3:
`# python Chapter\ 3\ -\ Statistial\ Experiments\ and\ Significance\ Testing.py`

Line 77

`print(np.mean(perm_diffs > mean_b - mean_a))`

blows up in Python 3.8.5, scipy 1.7.0, pandas 1.2.4, numpy 1.20.2.

In `ch_3_01_resampling.py`, changed it to

`print(np.mean(np.array(perm_diffs) > mean_b - mean_a))`


#### Chapter 6:
(On Mac M1 Arm processor)
```
$ arch -arm64 brew install libomp
$ sudo mkdir -p /usr/local/opt/libomp/lib
$ cd /usr/local/opt/libomp/lib
$ ln -s /opt/homebrew/lib/libomp.dylib libomp.dylib
$ pip install xgboost
$ python Chapter\ 6\ -\ Statistical\ Machine\ Learning.py
xgboost.core.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed (vcomp140.dll or libgomp-1.dll for Windows, libomp.dylib for Mac OSX, libgomp.so for Linux and other UNIX-like OSes). Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
  * You are running 32-bit Python on a 64-bit OS
 Error message(s): ['dlopen(/opt/homebrew/anaconda3/lib/python3.8/site-packages/xgboost/lib/libxgboost.dylib, 6): Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib\n  Referenced from: /opt/homebrew/anaconda3/lib/python3.8/site-packages/xgboost/lib/libxgboost.dylib\n  
 Reason: no suitable image found.  
 Did find:\n\t/usr/local/opt/libomp/lib/libomp.dylib: mach-o, but wrong architecture\n\t/opt/homebrew/Cellar/libomp/12.0.0/lib/libomp.dylib: mach-o, but wrong architecture']
```
I solved this problem by simply copying **libomp.a** and **libomp.dylib** from `/usr/local/Cellar/libomp/11.0.0/lib` on my Mac Intel Macbook Air, into `/opt/homebrew/Cellar/libomp/12.0.0/lib` on my Mac Arm M1 Macbook Pro. Now, Chapter 6, including all the `ch_6_*.py` files, run on the Arm M1 box. `brew install cmake` on the Intel Mac created these files originally.


(On Mac 64-bit Intel processor)
```
$ brew install cmake
$ pip install xgboost
```

#### Chapter 7:
```
$ pip install prince
$ python Chapter\ 7\ -\ Unsupervised\ Learning.py
...
4 : GOOGL
Traceback (most recent call last):
  File "Chapter 7 - Unsupervised Learning.py", line 213, in <module>
    ax = sns.scatterplot(x='XOM', y='CVX', hue=colors, style=colors,
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/_decorators.py", line 46, in inner_f
    return f(**kwargs)
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/relational.py", line 794, in scatterplot
    p = _ScatterPlotter(
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/relational.py", line 580, in __init__
    super().__init__(data=data, variables=variables)
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/_core.py", line 604, in __init__
    self.assign_variables(data, variables)
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/_core.py", line 667, in assign_variables
    plot_data, variables = self._assign_variables_longform(
  File "/opt/homebrew/anaconda3/lib/python3.8/site-packages/seaborn/_core.py", line 895, in _assign_variables_longform
    if val is not None and len(data) != len(val):
TypeError: object of type 'float' has no len()
```
See comment on line 32 of `ch_7_04_model_based_clustering_mixtures_of_normals_selecting_the_number_of_clusters.py`


Finally, a pedantic correction in the book.
First edition, page 194. "For each additional year that the worker is exposed to cotton dust the worker's PEFR measurement is reduced by -4.185." No, it's reduced by +4.185.
