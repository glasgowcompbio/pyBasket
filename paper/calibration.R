library(entropy)
library(rjags)
library(mvtnorm)
library(boot) # for inverse logit function
library(nonpar) # for Cochran's Q test
library(distrEx) #for TotalVarD
library(nonpar) # for function cochrans.q

###### General Simulation Settings:
K=6 # number of indications
alpha=0.1 # significance level for the test
num.sim=5000 # the number of simulations per simulation setting
Ni=24 # the maximum of total sample size for each indication group
Ni1=14 # stage-one sample size for each indication group
nik=matrix(NA,2,K) # each row records the number of patients in indication k at stage i
rik=matrix(NA,2,K) # each row records the number of responders in indication k at stage i
nik[1,]=rep(Ni1,K) # number of patients enrolled at stage 1

q0=0.2 # standard of care (null) response rate
q1=0.4 # target response rate
##################################################
############### Calibration Stage: ###############
##################################################
p0=rep(q0,K) # true rr: set the true rr for all indications to null rr (null scenario)
############## calibrate the tuning parameters for each method so that 
############## the type I error rate is well controlled under the null scenario

########## Calibration for CBHM with B distance measure & expnential correlation function:
Qf=0.05 # probability cut-off for interim analysis

########## Calibration for Independent Analysis:
posterior.ind=matrix(0,num.sim,K)
for (sim in 1:num.sim)
{
  ##### Stage 1:
  rik[1,]=sapply(1:K,FUN=function(x){rbinom(n=1,size=nik[1,x],prob=p0[x])})
  ############ Jags model for BHM:
  jags.data <- list("n"=nik[1,], "Y"=rik[1,], "K"=K)
  jags.fit <- jags.model(file = "independent.txt",data = jags.data,
                         n.adapt=1000,n.chains=1,quiet=T)
  update(jags.fit, 4000)
  independent.out <- coda.samples(jags.fit,variable.names = c("p"),n.iter=10000)
  ## Interim analysis:
  posterior=numeric()
  for (k in 1:K)
  {
    post.sample=independent.out[[1]][,paste("p[",k,"]",sep="")]
    posterior[k]=sum(post.sample>(q0+q1)/2)/length(post.sample)
  }
  ## Futility stop:
  stage2.stop=which(posterior<Qf)
  stage2.cont=which(posterior>=Qf)
  #nik[2,]=sapply(1:K,FUN=function(x){ifelse(is.element(x,stage2.cont),(Ni-Ni1)*K/length(stage2.cont),0)})
  nik[2,]=sapply(1:K,FUN=function(x){ifelse(is.element(x,stage2.cont),Ni-Ni1,0)})
  posterior.ind[sim,stage2.stop]=posterior[stage2.stop]
  ## Stage 2:
  if (length(stage2.cont)>0)
  {
    rik[2,]=sapply(1:K,FUN=function(x){rbinom(n=1,size=nik[2,x],prob=p0[x])})
    ri=colSums(as.matrix(rik[,stage2.cont]))
    ristar=ri
    ni=colSums(as.matrix(nik[,stage2.cont]))
    K1=length(stage2.cont)
    ############ Jags model for BHM:
    jags.data <- list("n"=ni, "Y"=ri, "K"=K1)
    jags.fit <- jags.model(file = "independent.txt",data = jags.data,
                           n.adapt=1000,n.chains=1,quiet=T)
    update(jags.fit, 4000)
    independent.out <- coda.samples(jags.fit,variable.names = c("p"),n.iter=10000)
    ## Final decision:
    posterior=numeric()
    if (K1==1)
    {
      post.sample=independent.out[[1]][,"p"]
      posterior=sum(post.sample>q0)/length(post.sample)
    }
    if (K1>1)
    {
      for (k in 1:K1)
      {
        post.sample=independent.out[[1]][,paste("p[",k,"]",sep="")]
        posterior[k]=sum(post.sample>q0)/length(post.sample)
      }
    }
    posterior.ind[sim,stage2.cont]=posterior
  }
  print(sim)
}
Q.independent=quantile(posterior.ind,1-alpha) 