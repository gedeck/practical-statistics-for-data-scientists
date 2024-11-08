## Practical Statistics for Data Scientists (R)
## Chapter 4. Regression and Prediction
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

# Import required R packages.

library(MASS)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(splines)
library(mgcv)

# Define paths to data sets. If you don't keep your data in the same directory as the code, adapt the path names.

PSDS_PATH <- file.path(dirname(dirname(getwd())))


lung <- read.csv(file.path(PSDS_PATH, 'data', 'LungDisease.csv'))
house <- read.csv(file.path(PSDS_PATH, 'data', 'house_sales.csv'), sep='\t')

## Simple Linear Regression
### The Regression Equation

plot(lung$Exposure, lung$PEFR, xlab="Exposure", ylab="PEFR")

model <- lm(PEFR ~ Exposure, data=lung)
model

plot(lung$Exposure, lung$PEFR, xlab="Exposure", ylab="PEFR", ylim=c(300,450), type="n", xaxs="i")
abline(a=model$coefficients[1], b=model$coefficients[2], col="blue", lwd=2)
text(x=.3, y=model$coefficients[1], labels=expression("b"[0]),  adj=0, cex=1.5)
x <- c(7.5, 17.5)
y <- predict(model, newdata=data.frame(Exposure=x))
segments(x[1], y[2], x[2], y[2] , col="red", lwd=2, lty=2)
segments(x[1], y[1], x[1], y[2] , col="red", lwd=2, lty=2)
text(x[1], mean(y), labels=expression(Delta~Y), pos=2, cex=1.5)
text(mean(x), y[2], labels=expression(Delta~X), pos=1, cex=1.5)
text(mean(x), 400, labels=expression(b[1] == frac(Delta ~ Y, Delta ~ X)), cex=1.5)

### Fitted Values and Residuals

fitted <- predict(model)
resid <- residuals(model)

lung1 <- lung %>%
  mutate(Fitted=fitted,
         positive = PEFR>Fitted) %>%
  group_by(Exposure, positive) %>%
  summarize(PEFR_max = max(PEFR),
            PEFR_min = min(PEFR),
            Fitted = first(Fitted),
            .groups='keep') %>%
  ungroup() %>%
  mutate(PEFR = ifelse(positive, PEFR_max, PEFR_min)) %>%
  arrange(Exposure)

plot(lung$Exposure, lung$PEFR, xlab="Exposure", ylab="PEFR")
abline(a=model$coefficients[1], b=model$coefficients[2], col="blue", lwd=2)
segments(lung1$Exposure, lung1$PEFR, lung1$Exposure, lung1$Fitted, col="red", lty=3)

## Multiple linear regression

print(head(house[, c('AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms',
               'Bedrooms', 'BldgGrade')]))

house_lm <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
                 Bedrooms + BldgGrade,
               data=house, na.action=na.omit)

house_lm

### Assessing the Model

summary(house_lm)

### Model Selection and Stepwise Regression

house_full <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
                   Bedrooms + BldgGrade + PropertyType + NbrLivingUnits +
                   SqFtFinBasement + YrBuilt + YrRenovated + NewConstruction,
                 data=house, na.action=na.omit)

## Code snippet 4.8
step_lm <- stepAIC(house_full, direction="both")
step_lm

lm(AdjSalePrice ~  Bedrooms, data=house)

### Weighted regression

house$Year = year(house$DocumentDate)
house$Weight = house$Year - 2005

house_wt <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
                 Bedrooms + BldgGrade,
               data=house, weight=Weight, na.action=na.omit)
round(cbind(house_lm=house_lm$coefficients,
            house_wt=house_wt$coefficients), digits=3)

## Factor variables in regression
### Dummy Variables Representation

head(house[, 'PropertyType'])

prop_type_dummies <- model.matrix(~PropertyType -1, data=house)
head(prop_type_dummies)

lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
     Bedrooms +  BldgGrade + PropertyType, data=house)

### Factor Variables with many levels

table(house$ZipCode)

zip_groups <- house %>%
  mutate(resid = residuals(house_lm)) %>%
  group_by(ZipCode) %>%
  summarize(med_resid = median(resid),
            cnt = n()) %>%
  # sort the zip codes by the median residual
  arrange(med_resid) %>%
  mutate(cum_cnt = cumsum(cnt),
         ZipGroup = factor(ntile(cum_cnt, 5)))
house <- house %>%
  left_join(dplyr::select(zip_groups, ZipCode, ZipGroup), by='ZipCode')

table(zip_groups[c('ZipGroup')])

## Interpreting the Regression Equation
### Correlated predictors

# The results from the stepwise regression are.

step_lm$coefficients

update(step_lm, . ~ . -SqFtTotLiving - SqFtFinBasement - Bathrooms)

### Confounding variables

lm(AdjSalePrice ~  SqFtTotLiving + SqFtLot +
     Bathrooms + Bedrooms +
     BldgGrade + PropertyType + ZipGroup,
   data=house, na.action=na.omit)

### Interactions and Main Effects

lm(AdjSalePrice ~  SqFtTotLiving*ZipGroup + SqFtLot +
     Bathrooms + Bedrooms +
     BldgGrade + PropertyType,
   data=house, na.action=na.omit)

## Testing the Assumptions: Regression Diagnostics
### Outliers

house_98105 <- house[house$ZipCode == 98105,]
lm_98105 <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms +
                 Bedrooms + BldgGrade, data=house_98105)

summary(lm_98105)

sresid <- rstandard(lm_98105)
idx <- order(sresid, decreasing=FALSE)
sresid[idx[1]]
resid(lm_98105)[idx[1]]

house_98105[idx[1], c('AdjSalePrice', 'SqFtTotLiving', 'SqFtLot',
                      'Bathrooms', 'Bedrooms', 'BldgGrade')]

### Influential values

seed <- 11
set.seed(seed)
x <- rnorm(25)
y <- -x/5 + rnorm(25)
x[1] <- 8
y[1] <- 8

plot(x, y, xlab='', ylab='', pch=16)
model <- lm(y~x)
abline(a=model$coefficients[1], b=model$coefficients[2], col="blue", lwd=3)
model <- lm(y[-1]~x[-1])
abline(a=model$coefficients[1], b=model$coefficients[2], col="red", lwd=3, lty=2)

# influential observations
std_resid <- rstandard(lm_98105)
cooks_D <- cooks.distance(lm_98105)
hat_values <- hatvalues(lm_98105)
plot(subset(hat_values, cooks_D > 0.08), subset(std_resid, cooks_D > 0.08),
     xlab='hat_values', ylab='std_resid',
     cex=10*sqrt(subset(cooks_D, cooks_D > 0.08)), pch=16, col='lightgrey')
points(hat_values, std_resid, cex=10*sqrt(cooks_D))
abline(h=c(-2.5, 2.5), lty=2)

lm_98105_inf <- lm(AdjSalePrice ~ SqFtTotLiving + SqFtLot +
                     Bathrooms +  Bedrooms + BldgGrade,
                   subset=cooks_D<.08, data=house_98105)

### Heteroskedasticity, Non-Normality and Correlated Errors

df <- data.frame(
  resid = residuals(lm_98105),
  pred = predict(lm_98105))

graph <- ggplot(df, aes(pred, abs(resid))) +
  geom_point() +
  geom_smooth(formula=y~x, method='loess') +
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) +
  theme_bw()
graph

hist(std_resid, main='')

### Partial Residual Plots and Nonlinearity

terms <- predict(lm_98105, type='terms')
partial_resid <- resid(lm_98105) + terms

df <- data.frame(SqFtTotLiving = house_98105[, 'SqFtTotLiving'],
                 Terms = terms[, 'SqFtTotLiving'],
                 PartialResid = partial_resid[, 'SqFtTotLiving'])
graph <- ggplot(df, aes(SqFtTotLiving, PartialResid)) +
  geom_point(shape=1) + scale_shape(solid = FALSE) +
  geom_smooth(linetype=2, formula=y~x, method='loess') +
  geom_line(aes(SqFtTotLiving, Terms)) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
graph

### Polynomial and Spline Regression

lm_poly <- lm(AdjSalePrice ~  poly(SqFtTotLiving, 2) + SqFtLot +
                BldgGrade +  Bathrooms +  Bedrooms,
              data=house_98105)
terms <- predict(lm_poly, type='terms')
partial_resid <- resid(lm_poly) + terms
lm_poly

df <- data.frame(SqFtTotLiving = house_98105[, 'SqFtTotLiving'],
                 Terms = terms[, 1],
                 PartialResid = partial_resid[, 1])
graph <- ggplot(df, aes(SqFtTotLiving, PartialResid)) +
  geom_point(shape=1) + scale_shape(solid = FALSE) +
  geom_smooth(linetype=2, formula=y~x, method='loess') +
  geom_line(aes(SqFtTotLiving, Terms)) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
graph

### Splines

knots <- quantile(house_98105$SqFtTotLiving, p=c(.25, .5, .75))
lm_spline <- lm(AdjSalePrice ~ bs(SqFtTotLiving, knots=knots, degree=3) +  SqFtLot +
                  Bathrooms + Bedrooms + BldgGrade,  data=house_98105)
lm_spline

terms1 <- predict(lm_spline, type='terms')
partial_resid1 <- resid(lm_spline) + terms1

df1 <- data.frame(SqFtTotLiving = house_98105[, 'SqFtTotLiving'],
                 Terms = terms1[, 1],
                 PartialResid = partial_resid1[, 1])

graph <- ggplot(df1, aes(SqFtTotLiving, PartialResid)) +
  geom_point(shape=1) + scale_shape(solid = FALSE) +
  geom_smooth(linetype=2, formula=y~x, method='loess') +
  geom_line(aes(SqFtTotLiving, Terms)) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
graph

### Generalized Additive Models

lm_gam <- gam(AdjSalePrice ~ s(SqFtTotLiving) + SqFtLot +
              Bathrooms +  Bedrooms + BldgGrade,
              data=house_98105)
terms <- predict.gam(lm_gam, type='terms')
partial_resid <- resid(lm_gam) + terms
lm_gam

df <- data.frame(SqFtTotLiving = house_98105[, 'SqFtTotLiving'],
                 Terms = terms[, 5],
                 PartialResid = partial_resid[, 5])
graph <- ggplot(df, aes(SqFtTotLiving, PartialResid)) +
  geom_point(shape=1) + scale_shape(solid = FALSE) +
  geom_smooth(linetype=2, formula=y~x, method='loess') +
  geom_line(aes(SqFtTotLiving, Terms)) +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
graph

