#### MFI R BOOTCAMP

# Case study: let's perform logistic regression on a simulated dataset

library(tidyverse)
library(knitr)

set.seed(1729)

# Function definitions

logit <- function(p) {
  return( log(p/(1-p)) )
}

expit <- function(x) {
  return( 1/(1 + exp(-x)) )
}

norm <- function(x) {sqrt(sum(x^2))}


# Initializations and variable assignments 
# Better use "<-" instead of "=" (<- was used in the original S-PLUS, and Google's R style guide insists on it)

N <- 500 # number of observations
p <- 2 # number of covariates (excluding intercept)
beta <- c(-2, 2, 1) # true coefficients (beta0, beta1, beta2)

# Data structures
# There are several choices of data structures, depending on your application and preference. For data, it's usually best to pre-allocate an array (i.e., vector or list), fill it with data, and then create a data frame out of it afterwards. 

X <- matrix(0L, nrow=N, ncol=p) 
colnames(X) <- c("x1", "x2")

Y <- vector("numeric", length=N)

# Let's simulate some data

for (i in 1:N) {
  X[i,] <- c(rnorm(n=1, mean=1), 
             rbinom(n=1, size=1, prob=0.6))
  
  eta_i <- beta %*% c(1, X[i,]) # matrix multiplication -- sizes must conform!
  
  Y[i] <- rbinom(n=1, size=1, prob=expit(eta_i))
}

# Always vectorize, if you can

X2 <- cbind(1, rnorm(n=N, mean=1), rbinom(n=N, size=1, prob=0.6))
eta <- rowSums(sweep(X2, 2, beta, "*"))
Y2 <- sapply(X=eta, FUN= function(eta) {rbinom(n=1, size=1, prob=expit(eta))}) # technically the apply functions are loops and not vectorized -- use if you want to "loop hide"


myDat.frame <- data.frame(y=Y, x1=X[,1], x2=as.factor(X[,2]))
myDat.tib <- tibble(y=Y, x1=X[,1], x2=as.factor(X[,2]))

levels(myDat.frame$x2) <- list(A = "0", B = "1")
myDat.tib$x2 <- recode_factor(myDat.tib$x2, "0" = "A", "1" = "B")

summary(myDat.tib)

# We'll stick with the tibble going forward

myDat.train <- myDat.tib[1:(N/2),]
myDat.test <- myDat.tib[(N/2 + 1):N,]

myModel <- glm(y ~ x1 + x2, data=myDat.train, family=binomial(link="logit"))
summary(myModel)

beta_hat <- myModel$coefficients
norm_diff <- norm(beta - beta_hat)

preds <- predict.glm(myModel, newdata=myDat.test, type=c("response"))
myDat.test <- cbind(myDat.test, y_pred = ifelse(preds > 0.5, 1, 0))
MSE <- mean(abs(myDat.test$y - myDat.test$y_pred))

# Plot some stuff

expit_beta_true <- function(x) { expit(beta[1] + x*beta[2])}
expit_beta_hat <- function(x) { expit(beta_hat[1] + x*beta_hat[2])}

ggplot(data=myDat.test, aes(x1, col=x2)) + 
  geom_histogram(bins = 30) + 
  facet_wrap(~ x2) + 
  labs(title="Histograms of x1", subtitle="By Category") + 
  theme(legend.position = "none")

ggplot(data=filter(myDat.test, x2 == "A"), aes(x=x1, y=y)) + 
  geom_point(alpha = 0.5, col="brown") + 
  geom_smooth(method="glm", method.args = list(family = "binomial"), aes(col="beta_hat_test")) +
  stat_function(fun=expit_beta_true, aes(col="beta_true")) + 
  stat_function(fun=expit_beta_hat, aes(col="beta_hat")) + 
  labs(title="Logistic regression curves", subtitle="For Category A") + 
  theme(legend.position = "right")
  

## Finding beta_hat directly using numerical optimization

mloglik <- function(beta) { # minus the log-likelihood
  ll <- 0
  for (i in 1:(N/2)) {
    eta_i <- beta %*% c(1, X[i,])
    ll <- ll + log( expit(eta_i) )*Y[i] + log(1 - expit(eta_i) )*(1 - Y[i])
  }
  return(-ll)
}

mloglik_grad <- function(beta) {
  gr <- rep(0, times=p+1)
  
  for (i in 1:(N/2)) {
    eta_i <- beta %*% c(1, X[i,])
    
    gr[1] <- gr[1] + (Y[i] - expit(eta_i))
    
    for (d in 1:p) {
      gr[d+1] <- gr[d+1] + (Y[i] - expit(eta_i))*X[i,d]
    }
  }
  return(-gr)
}

beta_hat_2 <- optim(par=rep(0,3), fn=mloglik, method="Nelder-Mead")
beta_hat_2 <- beta_hat_2$par # pretty close!
norm_diff_2 <- norm(beta_hat_2 - beta)

beta_hat_3 <- optim(par=rep(0,3), fn=mloglik, gr=mloglik_grad, method="BFGS")
beta_hat_3 <- beta_hat_3$par # actually further away from the true value, but closer to what 'glm' (i.e., Newton-Raphson) gives; no surprise since BFGS is a quasi-Newton method, unlike Nelder-Mead
norm_diff_3 <- norm(beta_hat_3 - beta)



## working with packages

install.packages("glmnet")
library(glmnet)

rm(list = ls()) # clear environment
set.seed(1729)


N <- 500 # number of observations
p <- 10 # number of covariates (excluding intercept)
beta <- rnorm(n=p+1, mean=0, sd=4) # true coefficients (beta0, beta1, beta2)
ssq <- 2

X <- matrix(0L, nrow=N, ncol=p+1) 

Y <- vector("numeric", length=N)

for (i in 1:N) {
  X[i,] <- c(1, rchisq(n=p, df=3))
  Y[i] <- beta %*% X[i,] + rnorm(n=1, mean=0, sd=sqrt(ssq))
}

beta_MLR <- solve(t(X) %*% X) %*% t(X) %*% Y
beta_MLR <- as.vector(beta_MLR)

dat <- data.frame(y=Y, X)

myModel <- lm(y ~. + 0, data=dat) # no intercept! Since we already have our own
myModel2 <- lm(y ~., data=subset(dat, select=-c(X1))) # with intercept, but without our column of ones
beta_OLS <- myModel$coefficients

## Stochastic processes (eg, a Vasicek model)

## stolen from https://www.r-bloggers.com/2010/04/fun-with-the-vasicek-interest-rate-model/

## define model parameters
r0 <- 0.03
theta <- 0.10
k <- 0.3
beta <- 0.03

## simulate short rate paths
n <- 10    # MC simulation trials
T <- 10    # total time
m <- 200   # subintervals
dt <- T/m  # difference in time each subinterval

r <- matrix(0,m+1,n)  # matrix to hold short rate paths
r[1,] <- r0

for(j in 1:n){
  for(i in 2:(m+1)){
    dr <- k*(theta-r[i-1,j])*dt + beta*sqrt(dt)*rnorm(1,0,1)
    r[i,j] <- r[i-1,j] + dr
  }
} 

## plot paths
t <- seq(0, T, dt)
rT.expected <- theta + (r0-theta)*exp(-k*t)
rT.stdev <- sqrt( beta^2/(2*k)*(1-exp(-2*k*t)))
matplot(t, r[,1:10], type="l", lty=1, main="Short Rate Paths", ylab="rt") 
abline(h=theta, col="red", lty=2)
lines(t, rT.expected, lty=2) 
lines(t, rT.expected + 2*rT.stdev, lty=2) 
lines(t, rT.expected - 2*rT.stdev, lty=2) 
points(0,r0)