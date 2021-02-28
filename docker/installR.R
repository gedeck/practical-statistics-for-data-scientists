local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos=r)
})


install.packages('remotes')
library(remotes)
install_version('ascii', '2.1')

if (!require(vioplot)) install.packages('vioplot')
if (!require(corrplot)) install.packages('corrplot')
if (!require(gmodels)) install.packages('gmodels')
if (!require(matrixStats)) install.packages('matrixStats')

if (!require(lmPerm)) install.packages('lmPerm')
if (!require(pwr)) install.packages('pwr')

if (!require(FNN)) install.packages('FNN')
if (!require(DMwR)) install.packages('DMwR')

if (!require(xgboost)) install.packages('xgboost')

if (!require(ellipse)) install.packages('ellipse')
if (!require(mclust)) install.packages('mclust')
if (!require(ca)) install.packages('ca')
if (!require(ggrepel)) install.packages('ggrepel')

if (!require(klaR)) install.packages('klaR')
