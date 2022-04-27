## Practical Statistics for Data Scientists (R)
## Chapter 7. Unsupervised Learning
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

# Import required R packages.

library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(ellipse)
library(mclust)
library(cluster)
library(ca)

# Define paths to data sets. If you don't keep your data in the same directory as the code, adapt the path names.

PSDS_PATH <- file.path(dirname(dirname(getwd())))

sp500_px <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_data.csv.gz'), row.names=1)
sp500_sym <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_sectors.csv'), stringsAsFactors = FALSE)
loan_data <- read.csv(file.path(PSDS_PATH, 'data', 'loan_data.csv.gz'))
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))
housetasks <- read.csv(file.path(PSDS_PATH, 'data', 'housetasks.csv'), row.names=1)

## Principal Components Analysis
### A simple example

oil_px <- sp500_px[, c('CVX', 'XOM')]
pca <- princomp(oil_px)
pca$loadings

loadings <- pca$loadings
graph <- ggplot(data=oil_px, aes(x=CVX, y=XOM)) +
  geom_point(alpha=.3) +
  scale_shape_manual(values=c(46)) +
  stat_ellipse(type='norm', level=.99, color='grey25') +
  geom_abline(intercept = 0, slope = loadings[2,1]/loadings[1,1], color='grey25', linetype=2) +
  geom_abline(intercept = 0, slope = loadings[2,2]/loadings[1,2],  color='grey25', linetype=2) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0)) +
  coord_cartesian(xlim=c(-3, 3), ylim=c(-3, 3)) +
  theme_bw()
graph

### Interpreting principal components

syms <- c( 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
           'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
top_sp <- sp500_px[row.names(sp500_px)>='2011-01-01', syms]
sp_pca <- princomp(top_sp)
par(mar=c(6,3,0,0)+.1, las=2)
screeplot(sp_pca, main='')

loadings <- sp_pca$loadings[,1:5]
loadings <- as.data.frame(loadings)
loadings$Symbol <- row.names(loadings)
loadings <- gather(loadings, 'Component', 'Weight', -Symbol)

loadings$Color = loadings$Weight > 0
graph <- ggplot(loadings, aes(x=Symbol, y=Weight, fill=Color)) +
  geom_bar(stat='identity', position='identity', width=.75) + 
  facet_grid(Component ~ ., scales='free_y') +
  guides(fill='none') +
  ylab('Component Loading') +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_text(angle=90, vjust=0.5))
graph

### Correspondence analysis

ca_analysis <- ca(housetasks)
plot(ca_analysis)

if (!require(ggrepel)) {
    install.packages('ggrepel')
    library(ggrepel)
}

ca_analysis$sv ** 2
summary(ca_analysis)

contrib = ca_analysis$sv ** 2
contrib = contrib / sum(contrib)
colcoord = as.data.frame(ca_analysis$colcoord)
rowcoord = as.data.frame(ca_analysis$rowcoord)
coords = rbind(
    cbind(rowcoord, type='rowcoord'),
    cbind(colcoord, type='columns')
)
row.names(coords) <- gsub('_', ' ', row.names(coords))

graph <- ggplot(coords, aes(x=Dim1, y=Dim2, color=type, label=rownames(coords), shape=type)) +
    geom_hline(yintercept=0, linetype='dotted', color='#444444') + 
    geom_vline(xintercept = 0, linetype='dotted', color='#444444') +
    geom_point() +
    geom_text_repel() + 
    xlab(sprintf('Dimension 1 (%.1f%%)', 100 * contrib[1])) +
    ylab(sprintf('Dimension 2 (%.1f%%)', 100 * contrib[2])) +
    scale_color_manual(values = c('blue', 'red')) +
    theme_bw() +
    theme(legend.position = "none") 
graph

## K-Means Clustering
### A Simple Example

set.seed(1010103)
df <- sp500_px[row.names(sp500_px)>='2011-01-01', c('XOM', 'CVX')]
km <- kmeans(df, centers=4, nstart=1)

df$cluster <- factor(km$cluster)
head(df)

centers <- data.frame(cluster=factor(1:4), km$centers)
centers

graph <- ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point() +
  scale_shape_manual(values = c(1, 3, 2, 4),
                     guide = guide_legend(override.aes=aes(size=1))) + 
  geom_point(data=centers,  aes(x=XOM, y=CVX), size=2, stroke=2, color='black')  +
  theme_bw() +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0)) +
  coord_cartesian(xlim=c(-2, 2), ylim=c(-2.5, 2.5))

graph

### K-Means Algorithm
# The _scikit-learn_ algorithm is repeated 10 times by default (`n_init`), `max_iter` is used to control the number of iterations.

syms <- c( 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
           'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
df <- sp500_px[row.names(sp500_px) >= '2011-01-01', syms]

set.seed(10010)
km <- kmeans(df, centers=5, nstart=10)

### Interpreting the Clusters

km$size

centers <- as.data.frame(t(km$centers))
names(centers) <- paste('Cluster', 1:5)
centers$Symbol <- row.names(centers)
centers <- gather(centers, 'Cluster', 'Mean', -Symbol)

centers$Color = centers$Mean > 0
graph <- ggplot(centers, aes(x=Symbol, y=Mean, fill=Color)) +
  geom_bar(stat='identity', position='identity', width=.75) + 
  facet_grid(Cluster ~ ., scales='free_y') +
  guides(fill='none')  +
  ylab('Component Loading') +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_text(angle=90, vjust=0.5))
graph

### Selecting the Number of Clusters

pct_var <- data.frame(pct_var = 0,
                      num_clusters=2:14)
totalss <- kmeans(df, centers=14, nstart=50, iter.max=100)$totss
for (i in 2:14) {
  kmCluster <- kmeans(df, centers=i, nstart=50, iter.max = 100)
  pct_var[i-1, 'pct_var'] <- kmCluster$betweenss / totalss
}

graph <- ggplot(pct_var, aes(x=num_clusters, y=pct_var)) +
  geom_line() +
  geom_point() +
  labs(y='% Variance Explained', x='Number of Clusters') +
  scale_x_continuous(breaks=seq(2, 14, by=2))   +
  theme_bw()
graph

## Hierarchical Clustering
### A Simple Example

syms1 <- c('GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 
           'XOM', 'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP',
           'WMT', 'TGT', 'HD', 'COST')

df <- sp500_px[row.names(sp500_px) >= '2011-01-01', syms1]
d <- dist(t(df))
hcl <- hclust(d)

### The Dendrogram

plot(hcl, ylab='distance', xlab='', sub='', main='')

cutree(hcl, k=4)

### Measures of Dissimilarity

cluster_fun <- function(df, method)
{
  d <- dist(df)
  hcl <- hclust(d, method=method)
  tree <- cutree(hcl, k=4)
  df$cluster <- factor(tree)
  df$method <- method
  return(df)
}

df0 <- sp500_px[row.names(sp500_px) >= '2011-01-01', c('XOM', 'CVX')]
df <- rbind(cluster_fun(df0, method='single'),
            cluster_fun(df0, method='average'),
            cluster_fun(df0, method='complete'),
            cluster_fun(df0, method='ward.D'))
df$method <- ordered(df$method, c('single', 'average', 'complete', 'ward.D'))

graph <- ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point(alpha=0.6) +
  scale_shape_manual(values=c(1, 3, 4, 2),
                     guide=guide_legend(override.aes=aes(size=2))) +
  facet_wrap( ~ method) +
  theme_bw()
graph

## Model based clustering
### Multivariate Normal Distribution
# > Define a colormap that corresponds to the probability levels

mu <- c(.5, -.5)
sigma <- matrix(c(1, 1, 1, 2), nrow=2)
prob <- c(.5, .75, .95, .99) ## or whatever you want
names(prob) <- prob ## to get id column in result
x <- NULL
for (p in prob){
  x <- rbind(x,  ellipse(x=sigma, centre=mu, level=p))
}
df <- data.frame(x, prob=factor(rep(prob, rep(100, length(prob)))))
names(df) <- c('X', 'Y', 'Prob')

## Figure 7-9: Multivariate normal ellipses
dfmu <- data.frame(X=mu[1], Y=mu[2])

graph <- ggplot(df, aes(X, Y)) + 
  geom_path(aes(linetype=Prob)) +
  geom_point(data=dfmu, aes(X, Y), size=3) +
  theme_bw()
graph

### Mixtures of Normals

df <- sp500_px[row.names(sp500_px)>='2011-01-01', c('XOM', 'CVX')]
mcl <- Mclust(df)
summary(mcl)

cluster <- factor(predict(mcl)$classification)
graph <- ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point(alpha=.8) +
  theme_bw() +
  scale_shape_manual(values = c(1, 3),
                     guide = guide_legend(override.aes=aes(size=2))) 
graph

summary(mcl, parameters=TRUE)$mean
summary(mcl, parameters=TRUE)$variance[,,1]
summary(mcl, parameters=TRUE)$variance[,,2]

### Selecting the number of clusters

plot(mcl, what='BIC', ask=FALSE, cex=.75)

## Scaling and Categorical Variables
### Scaling the Variables

defaults <- loan_data[loan_data$outcome=='default',]
df <- defaults[, c('loan_amnt', 'annual_inc', 'revol_bal', 'open_acc', 'dti', 'revol_util')]
km <- kmeans(df, centers=4, nstart=10)
centers <- data.frame(size=km$size, km$centers) 
print(round(centers, digits=2))

df0 <- scale(df)
km0 <- kmeans(df0, centers=4, nstart=10)
centers0 <- scale(km0$centers, center=FALSE, scale=1/attr(df0, 'scaled:scale'))
centers0 <- scale(centers0, center=-attr(df0, 'scaled:center'), scale=FALSE)
centers0 <- data.frame(size=km0$size, centers0) 
print(round(centers0, digits=2))

km <- kmeans(df, centers=4, nstart=10)
centers <- data.frame(size=km$size, km$centers) 
round(centers, digits=2)

### Dominant Variables

syms <- c('GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 
          'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
top_15 <- sp500_px[row.names(sp500_px)>='2011-01-01', syms]
sp_pca1 <- princomp(top_15)

screeplot(sp_pca1, main='')

round(sp_pca1$loadings[,1:2], 3)

### Categorical Data and Gower's Distance

x <- loan_data[1:5, c('dti', 'payment_inc_ratio', 'home_', 'purpose_')] %>%
  mutate(home_=as.factor(home_), purpose_=as.factor(purpose_))
x

daisy(x, metric='gower')

set.seed(301)
df <- loan_data[sample(nrow(loan_data), 250),
                c('dti', 'payment_inc_ratio', 'home_', 'purpose_')] %>%
  mutate(home_=as.factor(home_), purpose_=as.factor(purpose_))
d = daisy(df, metric='gower')
hcl <- hclust(d)
dnd <- as.dendrogram(hcl)
plot(dnd, leaflab='none', ylab='distance')

dnd_cut <- cut(dnd, h=.5)
df[labels(dnd_cut$lower[[4]]),]

### Problems with Clustering Mixed Data

df <- model.matrix(~ -1 + dti + payment_inc_ratio + home_ + pub_rec_zero, data=defaults)
df0 <- scale(df)
km0 <- kmeans(df0, centers=4, nstart=10)
centers0 <- scale(km0$centers, center=FALSE, scale=1/attr(df0, 'scaled:scale'))
round(scale(centers0, center=-attr(df0, 'scaled:center'), scale=FALSE), 2)
