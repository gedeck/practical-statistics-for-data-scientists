![Book cover](/images/cover-small.jpg)

# Code repository
**Practical Statistics for Data Scientists:**  

50+ Essential Concepts Using R and Python

by Peter Bruce, Andrew Bruce, and [Peter Gedeck](https://www.amazon.com/Peter-Gedeck/e/B082BJZJKX/)

- Publisher: O'Reilly Media; 2 edition (June 9, 2020)
- ISBN-13: 978-1492072942
- Buy on [Amazon](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/149207294X)
- Errata: http://oreilly.com/catalog/errata.csp?isbn=9781492072942

## R
Run the following commands in R to install all required packages
```
if (!require(vioplot)) install.packages('vioplot')
if (!require(corrplot)) install.packages('corrplot')
if (!require(gmodels)) install.packages('gmodels')
if (!require(matrixStats)) install.packages('matrixStats')

if (!require(lmPerm)) install.packages('lmPerm')
if (!require(pwr)) install.packages('pwr')

if (!require(FNN)) install.packages('FNN')
if (!require(klaR)) install.packages('klaR')
if (!require(DMwR)) install.packages('DMwR')

if (!require(xgboost)) install.packages('xgboost')

if (!require(ellipse)) install.packages('ellipse')
if (!require(mclust)) install.packages('mclust')
if (!require(ca)) install.packages('ca')
```

## Python
We recommend to use a conda environment to run the Python code. 
```
conda create -n sfds python
conda activate sfds

pip install jupyter
pip install pandas
pip install matplotlib
pip install scipy
pip install statsmodels
pip install wquantiles
pip install seaborn
pip install scikit-learn
pip install pygam
pip install dmba
pip install pydotplus

pip install imbalanced-learn
pip install prince

conda install --yes -c conda-forge xgboost
conda install --yes graphviz
```

## See also
- O'Reilly: https://oreil.ly/practicalStats_dataSci_2e
- Errata: http://oreilly.com/catalog/errata.csp?isbn=9781492072942
- The code repository for the first edition is at: https://github.com/andrewgbruce/statistics-for-data-scientists
