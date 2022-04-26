## Practical Statistics for Data Scientists (R)
## Chapter 1. Exploratory Data Analysis
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

# Import required packages.

library(dplyr)
library(tidyr)
library(ggplot2)
library(vioplot)
library(corrplot)
library(gmodels)
library(matrixStats)

# Import the datasets needed for chapter 1

PSDS_PATH <- file.path(dirname(dirname(getwd())))
 
state <- read.csv(file.path(PSDS_PATH, 'data', 'state.csv'))
dfw <- read.csv(file.path(PSDS_PATH, 'data', 'dfw_airline.csv'))
sp500_px <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_data.csv.gz'), row.names=1)
sp500_sym <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_sectors.csv'), stringsAsFactors = FALSE)
kc_tax <- read.csv(file.path(PSDS_PATH, 'data', 'kc_tax.csv.gz'))
lc_loans <- read.csv(file.path(PSDS_PATH, 'data', 'lc_loans.csv'))
airline_stats <- read.csv(file.path(PSDS_PATH, 'data', 'airline_stats.csv'), stringsAsFactors = FALSE)
airline_stats$airline <- ordered(airline_stats$airline, 
                                 levels=c('Alaska', 'American', 'Jet Blue', 'Delta', 'United', 'Southwest'))

## Estimates of Location
### Example: Location Estimates of Population and Murder Rates

# Table 1-2
state_asc <- state
state_asc[['Population']] <- formatC(state_asc[['Population']], format='d', digits=0, big.mark=',')
state_asc[1:8,]

mean(state[['Population']])
mean(state[['Population']], trim=0.1)
median(state[['Population']])

weighted.mean(state[['Murder.Rate']], w=state[['Population']])
library('matrixStats')
weightedMedian(state[['Murder.Rate']], w=state[['Population']])

## Estimates of Variability

sd(state[['Population']])
IQR(state[['Population']])
mad(state[['Population']])

### Percentiles and Boxplots

quantile(state[['Murder.Rate']], p=c(.05, .25, .5, .75, .95))

boxplot(state[['Population']]/1000000, ylab='Population (millions)')

### Frequency Table and Histograms

breaks <- seq(from=min(state[['Population']]), 
              to=max(state[['Population']]), length=11)
pop_freq <- cut(state[['Population']], breaks=breaks, 
                right=TRUE, include.lowest=TRUE)
state['PopFreq'] <- pop_freq
table(pop_freq)

options(scipen=5)
hist(state[['Population']], breaks=breaks)

### Density Estimates
# Density is an alternative to histograms that can provide more insight into the distribution of the data points.

hist(state[['Murder.Rate']], freq=FALSE )
lines(density(state[['Murder.Rate']]), lwd=3, col='blue')

## Exploring Binary and Categorical Data

barplot(as.matrix(dfw) / 6, cex.axis=0.8, cex.names=0.7, 
        xlab='Cause of delay', ylab='Count')

## Correlation

telecom <- sp500_px[, sp500_sym[sp500_sym$sector == 'telecommunications_services', 'symbol']]
telecom <- telecom[row.names(telecom) > '2012-07-01',]
telecom_cor <- cor(telecom)

# Next we focus on funds traded on major exchanges (sector == 'etf').

etfs <- sp500_px[row.names(sp500_px) > '2012-07-01', 
                 sp500_sym[sp500_sym$sector == 'etf', 'symbol']]
corrplot(cor(etfs), method='ellipse')

### Scatterplots

# plot(telecom$T, telecom$VZ, xlab='T', ylab='VZ', cex=.8)
plot(telecom$T, telecom$VZ, xlab='ATT (T)', ylab='Verizon (VZ)')
abline(h=0, v=0, col='grey')
dim(telecom)

## Exploring Two or More Variables
# Load the kc_tax dataset and filter based on a variety of criteria

kc_tax0 <- subset(kc_tax, TaxAssessedValue < 750000 & 
                  SqFtTotLiving > 100 &
                  SqFtTotLiving < 3500)
nrow(kc_tax0)

### Hexagonal binning and Contours 
#### Plotting numeric versus numeric data

# If the number of data points gets large, scatter plots will no longer be meaningful. Here methods that visualize densities are more useful. The `stat_hexbin` method for is one powerful approach.

graph <- ggplot(kc_tax0, (aes(x=SqFtTotLiving, y=TaxAssessedValue))) + 
  stat_binhex(color='white') + 
  theme_bw() + 
  scale_fill_gradient(low='white', high='blue') +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(x='Finished Square Feet', y='Tax-Assessed Value')
graph

# Visualize as a two-dimensional extension of the density plot.

graph <- ggplot(kc_tax0, aes(SqFtTotLiving, TaxAssessedValue)) +
  theme_bw() + 
  geom_point(color='blue', alpha=0.1) + 
  geom_density2d(color='white') + 
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(x='Finished Square Feet', y='Tax-Assessed Value')
graph

### Two Categorical Variables
# Load the `lc_loans` dataset

x_tab <- CrossTable(lc_loans$grade, lc_loans$status, 
                    prop.c=FALSE, prop.chisq=FALSE, prop.t=FALSE)

### Categorical and Numeric Data
# Boxplots of a column can be grouped by a different column.

boxplot(pct_carrier_delay ~ airline, data=airline_stats, ylim=c(0, 50), 
        cex.axis=.6, ylab='Daily % of Delayed Flights')

# Variation of boxplots called _violinplot_.

graph <- ggplot(data=airline_stats, aes(airline, pct_carrier_delay)) + 
  geom_violin(draw_quantiles = c(.25,.5,.75), linetype=2) +
  geom_violin(fill=NA, size=1.1) +
  coord_cartesian(ylim=c(0, 50)) +
  labs(x='', y='Daily % of Delayed Flights') +
  theme_bw()
graph

### Visualizing Multiple Variables

graph <- ggplot(subset(kc_tax0, ZipCode %in% c(98188, 98105, 98108, 98126)),
                aes(x=SqFtTotLiving, y=TaxAssessedValue)) + 
  stat_binhex(colour='white') + 
  theme_bw() + 
  scale_fill_gradient(low='gray95', high='black') +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(x='Finished Square Feet', y='Tax-Assessed Value') +
  facet_wrap('ZipCode')
graph
