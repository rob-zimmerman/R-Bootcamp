##### MFI R BOOTCAMP 2022

# Case study: let's perform logistic regression on a simulated dataset

library(tidyverse)
library(knitr)

set.seed(1729)

# Function definitions

logit <- function(p) {
  return( log(p/(1-p)) )
}

expit <- function(x) {
  return( 1/(1 + exp(-x)))
}

norm <- function(x) {
  sqrt(sum(x^2))
}


# Initializations and variable assignments
# Better use "<-" instead "=" (for reasons)

N <- 500 # number of observations
p <- 2 # number of covariates (excluding intercept!)
beta <- c(-2, 2, 1) # true coefficients (beta0, beta1, beta2)

# Data structures
# There are several choices of data structures, depending on your application (and personal preference). For data, it's usually best to pre-allocate an array (i.e., a vector or list), fill it in with data, and then create a data frame out of it afterwards.

X <- matrix(data=0L, nrow=N, ncol=p)
colnames(X) <- c("x1", "x2")

Y <- vector(mode="numeric", length=N)

# Let's simulate some data

for (i in 1:N) {
  X[i,] <- c( rnorm(n=1, mean=1), rbinom(n=1, size=1, prob=0.6) )
  
  eta_i <- beta %*% c(1, X[i,])
  
  Y[i] <- rbinom(n=1, size=1, prob=expit(eta_i))
}

# Always vectorize, if you can!

X2 <- cbind(1, rnorm(n=N, mean=1), rbinom(n=N, size=1, prob=0.6))
eta <- rowSums(sweep(X2, 2, beta, "*"))
Y2 <- sapply(X=eta, FUN= function(eta) { rbinom(n=1, size=1, prob=expit(eta))})

myDat.frame <- data.frame(y=Y, x1=X[,1], x2=as.factor(X[,2]))
myDat.tib <- tibble(y=Y, x1=X[,1], x2=as.factor(X[,2]))

levels(myDat.frame$x2) <- list(A = "0", B = "1")
myDat.tib$x2 <- recode_factor(myDat.tib$x2, "0" = "A", "1" = "B")

# We'll stick with the tibble going forward

myDat.train <- myDat.tib[1:(N/2), ] # training dataset
myDat.test <- myDat.tib[(N/2 + 1):N, ] # test (validation) dataset

myModel <- glm(y ~ x1 + x2, data=myDat.train, family=binomial(link="logit"))
summary(myModel)

beta_hat <- myModel$coefficients
norm_diff <- norm(beta - beta_hat)

preds <- predict.glm(object=myModel, newdata=myDat.test, type="response")
myDat.test <- cbind(myDat.test, y_pred = ifelse(preds > 0.5, yes=1, no=0))
MSE <- mean( (myDat.test$y - myDat.test$y_pred)^2 )

# Plot some stuff

expit_beta_true <- function(x) {expit(beta[1] + x*beta[2] + 0*beta[3])}
expit_beta_hat <- function(x) {expit(beta_hat[1] + x*beta_hat[2]  + 0*beta[3])}

ggplot(data=myDat.test, aes(x1, color=x2)) +
  geom_histogram(bins=30) +
  facet_wrap(~x2) +
  labs(title="Histograms of x1", subtitle="By category") +
  theme(legend.position = "none")

ggplot(data=filter(myDat.test, x2 == "A"), aes(x=x1, y=y)) +
  geom_point(alpha=0.5, col="brown") +
  stat_function(fun=expit_beta_true, aes(col="beta_true")) +
  stat_function(fun=expit_beta_hat, aes(col="beta_hat")) +
  geom_smooth(method="glm", method.args = list(family="binomial"), aes(col="beta_hat_gg")) +
  labs(title="Logistic regression curves", subtitle="For Category A") +
  theme(legend.position = "right")


# Let's find beta_hat directly using numerical optimization

mloglik <- function(beta) { # minus the log-likelihood
  ll <- 0
  for (i in 1:(N/2)) {
    eta_i <- beta %*% c(1, X[i,])
    ll <- ll + Y[i]*log( expit(eta_i) ) + (1 - Y[i])*log(1 - expit(eta_i) )
  }
  return(-ll)
}
  
mloglik_grad <- function(beta) { # gradient of minus the log-likelihood
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

beta_hat_2 <- optim(par=rep(0,times=3),
                    fn=mloglik,
                    method="Nelder-Mead")
beta_hat_2 <- beta_hat_2$par
norm_diff_2 <- norm(beta_hat_2 - beta_hat)
  
  
  
mloglik_b1 <- function(beta1) {mloglik(c(0, beta1, 0))}

optimize(      f=mloglik_b1,
      interval=c(-1,1))




### Linear regression

rm(list = ls()) # clear environment
set.seed(1729)  

N <- 500 # number of observations
p <- 10 # number of covariates

beta <- rnorm(n=p+1, mean=0, sd=4) # true coefficients (beta0, beta1, ..., beta_p)
ssq <- 2

X <- matrix(0L, nrow=N, ncol=p+1)

Y <- rep(0, times=N)

for (i in 1:N) {
  X[i,] <- c(1, rchisq(n=p, df=3))
  Y[i] <- beta %*% X[i,] + rnorm(n=1, mean=0, sd=sqrt(ssq))
}

beta_MLR <- solve( t(X) %*% X ) %*% t(X) %*% Y
beta_MLR <- as.vector(beta_MLR)

dat <- data.frame(y=Y, X=X)

myModel0 <- lm(y ~., dat=dat)
myModel <- lm(y ~. + 0, data=dat) # no intercept! If we want to use our own estimate
myModel2 <- lm(y ~., data=subset(dat, select=-c(X.1))) # with intercept, but without our column of ones
summary(myModel0)
summary(myModel)
summary(myModel2)
beta_OLS <- myModel$coefficients

Y_pred <- beta_OLS %*% t(X)
Y_pred <- as.vector(Y_pred)

MSE_preds <- norm(Y_pred - Y)
MSE_beta <- norm(beta_OLS - beta)


## Stochastic processes (eg, Vasicek model)

# Define some model parameters
r0 <- 0.03
theta <- 0.10
k <- 0.3
sigma <- 0.03

# Simulate some short rate paths
n <- 10 # number of simulations
TT <- 10 # time
m <- 200 # number of sub-intervals
dt <- TT/m # difference in time in each sub-interval

r <- matrix(0L, nrow=m+1, ncol=n)
r[1,] <- r0

for (j in 1:n) {
  for (i in 2:(m+1)) {
    dr <- k*(theta - r[i-1, j])*dt + sigma*sqrt(dt)*rnorm(n=1, mean=0, sd=1)
    r[i,j] <- r[i-1, j] + dr
  }
}

# Plot our interest rate paths along with a confidence band
t <- seq(from=0, to=TT, by=dt)
rT.expected <- theta + (r0 - theta)*exp(-k*t)
rT.stdev <- sqrt( sigma^2/(2*k)*(1 - exp(-2*k*t)) )
matplot(x=t, y=r, type="l", lty=1, main="Vasicek short rate paths", ylab="rt", xlab="t")
lines(t, rT.expected, lty=4)
lines(t, rT.expected + 1.96*rT.stdev, lty=2)
lines(t, rT.expected - 1.96*rT.stdev, lty=2)
points(0, r0)
