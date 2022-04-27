## Practical Statistics for Data Scientists (R)
## Chapter 5. Classification
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

# Import required R packages.

library(klaR)
library(MASS)
library(dplyr)
library(ggplot2)
library(FNN)
library(mgcv)
library(rpart)

# Define paths to data sets. If you don't keep your data in the same directory as the code, adapt the path names.

PSDS_PATH <- file.path(dirname(dirname(getwd())))

loan3000 <- read.csv(file.path(PSDS_PATH, 'data', 'loan3000.csv'), stringsAsFactors=TRUE)
loan_data <- read.csv(file.path(PSDS_PATH, 'data', 'loan_data.csv.gz'), stringsAsFactors=TRUE)
full_train_set <- read.csv(file.path(PSDS_PATH, 'data', 'full_train_set.csv.gz'), stringsAsFactors=TRUE)

# order the outcome variable
loan3000$outcome <- ordered(loan3000$outcome, levels=c('paid off', 'default'))
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))
full_train_set$outcome <- ordered(full_train_set$outcome, levels=c('paid off', 'default'))

## Naive Bayes
### The Naive Solution

naive_model <- NaiveBayes(outcome ~ purpose_ + home_ + emp_len_, 
                          data = na.omit(loan_data))
naive_model$table

new_loan <- loan_data[147, c('purpose_', 'home_', 'emp_len_')]
row.names(new_loan) <- NULL
new_loan

predict(naive_model, new_loan)

print(predict(naive_model, new_loan))

# Due to changes in R's default behavior of converting strings to factors, it is important to convert string columns to factors when using the naive Bayes implementation in `klaR`. Use either `stringsAsFactors=TRUE` when loading the data or convert to factors explicitly.

#### Example not in book

less_naive <- NaiveBayes(outcome ~ borrower_score + payment_inc_ratio + 
                           purpose_ + home_ + emp_len_, data = loan_data)
less_naive$table[1:2]

stats <- less_naive$table[[1]]
graph <- ggplot(data.frame(borrower_score=c(0,1)), aes(borrower_score)) +
  stat_function(fun = dnorm, color='blue', linetype=1, 
                args=list(mean=stats[1, 1], sd=stats[1, 2])) +
  stat_function(fun = dnorm, color='red', linetype=2, 
                args=list(mean=stats[2, 1], sd=stats[2, 2])) +
  labs(y='probability')
graph

## Discriminant Analysis
### A Simple Example

loan_lda <- lda(outcome ~ borrower_score + payment_inc_ratio,
                data=loan3000)
loan_lda$scaling

pred <- predict(loan_lda)
print(head(pred$posterior))

#### Figure 5.1

# pred <- predict(loan_lda)
# lda_df <- cbind(loan3000, prob_default=pred$posterior[,'default'])

# x <- seq(from=.33, to=.73, length=100)
# y <- seq(from=0, to=20, length=100)
# newdata <- data.frame(borrower_score=x, payment_inc_ratio=y)
# pred <- predict(loan_lda, newdata=newdata)
# lda_df0 <- cbind(newdata, outcome=pred$class)

# graph <- ggplot(data=lda_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
#   geom_point(alpha=.6) +
#   scale_color_gradient2(low='white', high='blue') +
#   scale_x_continuous(expand=c(0,0)) + 
#   scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
#   geom_line(data=lda_df0, col='darkgreen', size=2, alpha=.8) +
#   theme_bw()
# graph

# # [Book]
# graph
# dev.off()

pred <- predict(loan_lda)
lda_df <- cbind(loan3000, prob_default=pred$posterior[,'default'])

center <- 0.5 * (loan_lda$mean[1, ] + loan_lda$mean[2, ])
slope <- -loan_lda$scaling[1] / loan_lda$scaling[2]
intercept = center[2] - center[1] * slope

graph <- ggplot(data=lda_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
  geom_point(alpha=.6) +
  scale_color_gradientn(colors=c('#ca0020', '#f7f7f7', '#0571b0')) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  geom_abline(slope=slope, intercept=intercept, color='darkgreen') +
  theme_bw()

graph

## Logistic regression
### Logistic Response Function and Logit

logistic_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                        home_ + emp_len_ + borrower_score,
                      data=loan_data, family='binomial')
logistic_model
summary(logistic_model)

p <- seq(from=0.01, to=.99, by=.01)
df <- data.frame(p = p,
                 logit = log(p/(1-p)),
                 odds = p/(1-p))

graph <- ggplot(data=df, aes(x=p, y=logit)) +
  geom_line() +
  labs(x = 'p', y='logit(p)') +
  theme_bw()
graph

### Predicted Values from Logistic Regression

pred <- predict(logistic_model)
summary(pred)

prob <- 1/(1 + exp(-pred))
summary(prob)

### Interpreting the Coefficients and Odds Ratios

graph <- ggplot(data=df, aes(x=logit, y=odds)) +
  geom_line() +
  labs(x='log(odds ratio)', y='odds ratio') +
  coord_cartesian(xlim=c(0, 5), ylim=c(1, 100)) +
  theme_bw()
graph

### Logistic regression with splines

logistic_gam <- gam(outcome ~ s(payment_inc_ratio) + purpose_ + 
                      home_ + emp_len_ + s(borrower_score),
                    data=loan_data, family='binomial')
logistic_gam

### Assessing the Model¶

terms <- predict(logistic_gam, type='terms')
partial_resid <- resid(logistic_gam) + terms
df <- data.frame(payment_inc_ratio = loan_data[, 'payment_inc_ratio'],
                 terms = terms[, 's(payment_inc_ratio)'],
                 partial_resid = partial_resid[, 's(payment_inc_ratio)'])

graph <- ggplot(df, aes(x=payment_inc_ratio, y=partial_resid, solid = FALSE)) +
  geom_point(shape=46, alpha=0.4) +
  geom_line(aes(x=payment_inc_ratio, y=terms), 
            color='red', alpha=0.5, size=1.5) +
  labs(y='Partial Residual') +
  coord_cartesian(xlim=c(0, 25)) +
  theme_bw()
graph

df <- data.frame(payment_inc_ratio = loan_data[, 'payment_inc_ratio'],
                 terms = 1/(1 + exp(-terms[, 's(payment_inc_ratio)'])),
                 partial_resid = 1/(1 + exp(-partial_resid[, 's(payment_inc_ratio)'])))

graph <- ggplot(df, aes(x=payment_inc_ratio, y=partial_resid, solid = FALSE)) +
  geom_point(shape=46, alpha=0.4) +
  geom_line(aes(x=payment_inc_ratio, y=terms), 
            color='red', alpha=0.5, size=1.5) +
  labs(y='Partial Residual') +
  coord_cartesian(xlim=c(0, 25)) +
  theme_bw()
graph

## Evaluating Classification Models
### Confusion Matrix

# Confusion matrix
pred <- predict(logistic_gam, newdata=loan_data)
pred_y <- as.numeric(pred > 0)
true_y <- as.numeric(loan_data$outcome=='default')
true_pos <- (true_y==1) & (pred_y==1)
true_neg <- (true_y==0) & (pred_y==0)
false_pos <- (true_y==0) & (pred_y==1)
false_neg <- (true_y==1) & (pred_y==0)
conf_mat <- matrix(c(sum(true_pos), sum(false_pos),
                     sum(false_neg), sum(true_neg)), 2, 2)
colnames(conf_mat) <- c('Yhat = 1', 'Yhat = 0')
rownames(conf_mat) <- c('Y = 1', 'Y = 0')
conf_mat

### Precision, Recall, and Specificity

# precision
conf_mat[1, 1] / sum(conf_mat[,1])
# recall
conf_mat[1, 1] / sum(conf_mat[1,])
# specificity
conf_mat[2, 2] / sum(conf_mat[2,])

### ROC Curve

idx <- order(-pred)
recall <- cumsum(true_y[idx] == 1) / sum(true_y == 1)
specificity <- (sum(true_y == 0) - cumsum(true_y[idx] == 0)) / sum(true_y == 0)
roc_df <- data.frame(recall = recall, specificity = specificity)

graph <- ggplot(roc_df, aes(x=specificity, y=recall)) +
  geom_line(color='blue') + 
  scale_x_reverse(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) + 
  geom_line(data=data.frame(x=(0:100) / 100), aes(x=x, y=1-x),
            linetype='dotted', color='red') +
  theme_bw() + theme(plot.margin=unit(c(5.5, 10, 5.5, 5.5), "points"))
graph

### AUC

sum(roc_df$recall[-1] * diff(1-roc_df$specificity))
head(roc_df)

graph <- ggplot(roc_df, aes(specificity)) +
  geom_ribbon(aes(ymin=0, ymax=recall), fill='blue', alpha=.3) +
  scale_x_reverse(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) +
  labs(y='recall') +
  theme_bw() + theme(plot.margin=unit(c(5.5, 10, 5.5, 5.5), "points"))
graph

## Strategies for Imbalanced Data
### Undersampling

mean(full_train_set$outcome=='default')

full_model <- glm(outcome ~ payment_inc_ratio + purpose_ + home_ + 
                            emp_len_+ dti + revol_bal + revol_util,
                  data=full_train_set, family='binomial')
pred <- predict(full_model)
mean(pred > 0)

 mean(full_train_set$outcome=='default') / mean(pred > 0)

### Oversampling and Up/Down Weighting

wt <- ifelse(full_train_set$outcome=='default', 
             1 / mean(full_train_set$outcome == 'default'), 1)
full_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                  home_ + emp_len_+ dti + revol_bal + revol_util,
                  data=full_train_set, weight=wt, family='quasibinomial')
pred <- predict(full_model)
mean(pred > 0)

### Data Generation
# There are a variety of SMOTE implementation available in R. The package `unbalanced` provides SMOTE and other methods. Unfortunately, it's not working for your dataset. 
# 
# The SMOTE implementation in the package `DMwR` works. However `DMwR` is no longer supported.

loan_data_samp <- sample_frac(full_train_set, .05)

# # install.packages('unbalanced')
# library(unbalanced)
# head(full_train_set)
# smote_data <- ubSMOTE(loan_data_samp, loan_data_samp$outcome, 
#                       perc.over = 2000, k = 5, perc.under = 100)
# head(smote_data)

# install.packages('DMwR')
# library(DMwR)
# smote_data <- SMOTE(outcome ~ ., loan_data_samp, 
#                     perc.over = 2000, perc.under=100)
# dim(loan_data_samp)
# dim(smote_data)
# head(smote_data)

### Exploring the Predictions

loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, 
                   control = rpart.control(cp=.005))

geom_abline(slope=slope, intercept=intercept, color='darkgreen')
lda_pred <- data.frame(borrower_score = c((0 - intercept) / slope, (20 - intercept) / slope),
                       payment_inc_ratio = c(0, 20),
                       method = rep('LDA', 2))

tree_pred <- data.frame(borrower_score = c(0.375, 0.375, 0.475, 0.475, 0.575, 0.575),
                        payment_inc_ratio = c(0, 4.426,  4.426, 10.42, 10.42, 20),
                        method = rep('Tree', 6))

glm0 <- glm(outcome ~ (payment_inc_ratio) +  (borrower_score),
            data=loan3000, family='binomial')
y <- seq(from=0, to=20, length=100)
x <- (-glm0$coefficients[1] - glm0$coefficients[2]*y)/glm0$coefficients[3]
glm0_pred <- data.frame(borrower_score=x, payment_inc_ratio=y, method='Logistic')

gam1 <- gam(outcome ~ s(payment_inc_ratio) +  s(borrower_score),
            data=loan3000, family='binomial')
gam_fun <- function(x){
  rss <- sum(predict(gam1, newdata=data.frame(borrower_score=x, payment_inc_ratio=y))^2)
}
est_x <- nlminb(seq(from=.33, to=.73, length=100), gam_fun )
gam1_pred <- data.frame(borrower_score=est_x$par, payment_inc_ratio=y, method="GAM")

loan_fits <- rbind(lda_pred,
                   tree_pred,
                   glm0_pred,
                   gam1_pred)

graph <- ggplot(data=loan_fits, aes(x=borrower_score, y=payment_inc_ratio, color=method, linetype=method)) +
  geom_line(size=1.5) +
  theme(legend.key.width = unit(2,"cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 1)))
graph
