# Code repository
<table width='100%'>
 <tr>
  <td><img src='/images/OReilly-english.jpg' width=300></td>
  <td>
   <b>Practical Statistics for Data Scientists:</b>

50+ Essential Concepts Using R and Python

by Peter Bruce, Andrew Bruce, and [Peter Gedeck](https://www.amazon.com/Peter-Gedeck/e/B082BJZJKX/)

- Publisher: [O'Reilly Media](https://oreil.ly/practicalStats_dataSci_2e); 2 edition (June 9, 2020)
- ISBN-13: 978-1492072942
- Buy on [Amazon](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/149207294X)
- Errata: http://oreilly.com/catalog/errata.csp?isbn=9781492072942
    </td>
  </tr>
</table>
 

## Online
View the notebooks online:
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/gedeck/practical-statistics-for-data-scientists/tree/master/)

Excecute the notebooks in Binder: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gedeck/practical-statistics-for-data-scientists/HEAD)

 This can take some time if the binder environment needs to be rebuilt. 

## Other language versions
<table>
  <tr>
   <td><img src='/images/OReilly-english.jpg' width=200></td>
  <td><b>English:</b><br>
   Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python<br>
   2020: ISBN 149207294X<br>
   <a href='https://www.google.com/books/edition/Practical_Statistics_for_Data_Scientists/F2bcDwAAQBAJ?hl=en'>Google books</a>,
   <a href='https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/149207294X'>Amazon</a>
  </td>
 </tr>

 <tr>
  <td><img src='/images/OReilly-japanese.jpg' width=200></td>
  <td><b>Japanese:</b><br>
   データサイエンスのための統計学入門 第2版 ―予測、分類、統計モデリング、統計的機械学習とR/Pythonプログラミング <br>
   2020: ISBN 487311926X, 
   Shinya Ohashi (supervised), Toshiaki Kurokawa (translated)<br>
   <a href='https://www.google.com/books/edition/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E7%B5%B1/d7EJzgEACAAJ?hl=en'>Google books</a>,
   <a href='https://www.amazon.co.jp/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E7%B5%B1%E8%A8%88%E5%AD%A6%E5%85%A5%E9%96%80-%E2%80%95%E4%BA%88%E6%B8%AC%E3%80%81%E5%88%86%E9%A1%9E%E3%80%81%E7%B5%B1%E8%A8%88%E3%83%A2%E3%83%87%E3%83%AA%E3%83%B3%E3%82%B0%E3%80%81%E7%B5%B1%E8%A8%88%E7%9A%84%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%A8R-Python%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0-Peter-Bruce/dp/487311926X'>Amazon</a>
  </td>
 </tr>
 <tr>
  <td><img src='/images/OReilly-german.jpg' width=200></td>
  <td><b>German:</b><br>
   Praktische Statistik für Data Scientists: 50+ essenzielle Konzepte mit R und Python <br>
   2021: ISBN 3960091532, Marcus Fraaß (Übersetzer)<br>
   <a href='https://www.google.com/books/edition/Praktische_Statistik_f%C3%BCr_Data_Scientist/yeMCzgEACAAJ?hl=en'>Google books</a>,
   <a href='https://www.amazon.de/Praktische-Statistik-f%C3%BCr-Data-Scientists/dp/3960091532'>Amazon</a>
  </td>
 </tr>
 <tr>
  <td><img src='/images/OReilly-korean.jpg' width=200></td>
  <td><b>Korean:</b><br>
   Practical Statistics for Data Scientists: 데이터 과학을 위한 통계(2판) 
   2021: ISBN 9791162244180, Junyong Lee (translation)
   <br>
   <a href='https://www.google.com/books/edition/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EA%B3%BC%ED%95%99%EC%9D%84_%EC%9C%84%ED%95%9C_%ED%86%B5%EA%B3%84_2%ED%8C%90/9E9qzgEACAAJ?hl=en'>Google books</a>,
   <a href='https://www.hanbit.co.kr/store/books/look.php?p_code=B2862122581'>Hanbit media</a>
   
  </td>
 </tr>
</table>

## See also
- The code repository for the first edition is at: https://github.com/andrewgbruce/statistics-for-data-scientists


# Setup R and Python environments
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
conda env update -n sfds -f environment.yml
```



