library(entropy)
library(rjags)
library(mvtnorm)
library(boot) # for inverse logit function
library(nonpar) # for Cochran's Q test
library(distrEx) #for TotalVarD
library(nonpar) # for function cochrans.q
K=6 # the number of indications
num.sim=5000 # the number of simulations per setting
Ni=24 # the maximum of total sample size for each indication group
Ni1=14 # stage-one sample size for each indication group
nik=matrix(NA,2,K) # each row records the number of patients in indication k at stage i
rik=matrix(NA,2,K) # each row records the number of responders in indication k at stage i
nik[1,]=rep(Ni1,K) # number of patients enrolled at stage 1
q0=0.2 # standard of care (null) response rate
q1=0.4 # target response rate
###### p.scenario: each row stores the true rrs in one simulation setting
p.scenario = t(sapply(0:(K-1),function(x){c(rep(q1,x),rep(q0,K-x))}))
methods=c("Independent")

###### parameters
Qf=0.05 # probability cut-off for interim analysis
epsilon = 3*(q1-q0)/K # the small value added to the number of responsders for the indication groups that have equal sample rr
C=0.5 # threshold for futility stopping for Liu's two-stage method, follows Liu et al. (2017)

#### the following Qs are calibrated to have the same level of type I error control under the null scenario. 
#### Please see calibration.R for details.
Q.independent # the calibrated prob cut-off for final decision for independent analysis
#############################################
############## Full Simulations #############
#############################################

##################### Create simulated data: #####################
simdata=list()
for (scenario in 1:nrow(p.scenario))
{
  simdata[[scenario]]=matrix(NA,num.sim*Ni,K)
  p0=p.scenario[scenario,]
  for (sim in 1:num.sim)
  {
    simdata[[scenario]][((sim-1)*Ni+1):(sim*Ni),]=sapply(1:K,FUN=function(x){rbinom(n=Ni,size=1,prob=p0[x])})
  }
}

##################### Simulation: #####################
OC=list() # store operating charasteristics
Decision=list()
decisions=list() # save the testing results for all simulations, optional
Bias=list() # store absolute bias
MSE=list() # store MSE
samplesize=matrix(0,nrow(p.scenario),length(methods)) # average sample size considering interim stop
for (scenario in 1:nrow(p.scenario))
{
  p0=p.scenario[scenario,]
  # decision*: store the decision of each simulation. -1: interim stop, 0: do not reject, 1: reject
  decision.independent=matrix(NA,num.sim,K)
  Bias.independent=matrix(0,num.sim,K)
  pest.independent=matrix(0,num.sim,K)
  MSE.independent=matrix(0,num.sim,K)
  samplesize[scenario,]=rep(0,length(methods))
  Decision[[scenario]]=matrix(0,length(methods),5)
  colnames(Decision[[scenario]])=c("% Perfect","# TP","# TN","# FP","# FN")
  rownames(Decision[[scenario]])=methods
  tp=which(p0>=q1)
  tn=which(p0<q1)
  
  for (sim in 1:num.sim)
  {
    ########### Stage 1 data:
    stage1resp=simdata[[scenario]][((sim-1)*Ni+1):((sim-1)*Ni+Ni1),]
    #stage1resp=sapply(1:K,FUN=function(x){rbinom(n=nik[1,x],size=1,prob=p0[x])})
    rik[1,]=colSums(stage1resp)    
    rikstar=rik
    zero=rep(0,K)
      
    ############################## Independent analysis:
    ############ Jags model:
    jags.data <- list("n"=nik[1,], "Y"=rik[1,], "K"=K)
    jags.fit <- jags.model(file = "independent.txt",data = jags.data,
                           n.adapt=1000,n.chains=1,quiet=T)
    update(jags.fit, 4000)
    independent.out <- coda.samples(jags.fit,variable.names = c("p"),n.iter=10000)
    ### Interim analysis:
    posterior=numeric()
    for (k in 1:K)
    {
      post.sample=independent.out[[1]][,paste0("p[",k,"]")]
      posterior[k]=sum(post.sample>(q0+q1)/2)/length(post.sample)
    }
    ## Futility stop:
    stage2.stop=which(posterior<Qf)
    stage2.cont=which(posterior>=Qf)
    nik[2,]=sapply(1:K,FUN=function(x){ifelse(is.element(x,stage2.cont),Ni-Ni1,0)})
    decision.independent[sim,stage2.stop]=-1
    if(length(stage2.stop)>0)
    {
      Bias.independent[sim,stage2.stop]=abs(summary(independent.out[[1]])[[1]][sapply(1:length(stage2.stop),FUN=function(x){paste0("p[",stage2.stop[x],"]")}),"Mean"]-p0[stage2.stop])
      pest.independent[sim,stage2.stop]=summary(independent.out[[1]])[[1]][sapply(1:length(stage2.stop),FUN=function(x){paste0("p[",stage2.stop[x],"]")}),"Mean"]
    }
    ## Stage 2:
    if (length(stage2.cont)>0)
    {
      rik[2,]=colSums(simdata[[scenario]][((sim-1)*Ni+Ni1+1):((sim-1)*Ni+Ni),] %*% diag(ifelse(nik[2,]>0,1,0),K))
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
      
      ### Final decision:
      posterior=numeric()
      if (K1==1)
      {
        post.sample=independent.out[[1]][,"p"]
        posterior=sum(post.sample>q0)/length(post.sample)
        ##### Bias:
        Bias.independent[sim,stage2.cont]=abs(summary(independent.out[[1]])[[1]]["Mean"]-p0[stage2.cont])
        pest.independent[sim,stage2.cont]=summary(independent.out[[1]])[[1]]["Mean"]
        #MSE.independent[sim,stage2.cont]=Bias.independent[sim,stage2.cont]^2 + (summary(independent.out[[1]])[[1]][,"SD"])^2
      }
      if (K1>1)
      {
        for (k in 1:K1)
        {
          post.sample=independent.out[[1]][,paste0("p[",k,"]")]
          posterior[k]=sum(post.sample>q0)/length(post.sample)
        }
        ##### Bias:
        Bias.independent[sim,stage2.cont]=abs(summary(independent.out[[1]])[[1]][sapply(1:K1,FUN=function(x){paste0("p[",x,"]")}),"Mean"]-p0[stage2.cont])
        pest.independent[sim,stage2.cont]=summary(independent.out[[1]])[[1]][sapply(1:K1,FUN=function(x){paste0("p[",x,"]")}),"Mean"]
      }
      decision.independent[sim,stage2.cont]=ifelse(posterior>Q.independent,1,0)
    }
    ni.independent=sum(nik)
    
    # update average sample size
    samplesize[scenario,]=samplesize[scenario,]+c(ni.cbhm,ni.exnex,ni.liu,ni.bhm,ni.independent)
    
    # Summarize TP, TN, FP, FN:
    Decision[[scenario]]['Independent',"% Perfect"]=Decision[[scenario]]['Independent',"% Perfect"] + 1*(sum(c(decision.independent[sim,tp]==1,decision.independent[sim,tn]<=0))==K)
    Decision[[scenario]]['Independent',"# TP"]=Decision[[scenario]]['Independent',"# TP"] + sum(decision.independent[sim,tp]==1)
    Decision[[scenario]]['Independent',"# TN"]=Decision[[scenario]]['Independent',"# TN"] +  sum(decision.independent[sim,tn]<=0)
    Decision[[scenario]]['Independent',"# FP"]=Decision[[scenario]]['Independent',"# FP"] +  sum(decision.independent[sim,tn]==1)
    Decision[[scenario]]['Independent',"# FN"]=Decision[[scenario]]['Independent',"# FN"] +  sum(decision.independent[sim,tp]<=0)
    
    print(sim)
    # optional: save the results for all simulations
    #decisions[[scenario]]=list(decision.cbhm,decision.exnex,decision.liu,decision.bhm,decision.independent)
  }
  
  Decision[[scenario]]=Decision[[scenario]]/num.sim # average across all num.sim simulations
  
  OC[[scenario]]=matrix(NA,2*length(methods),K)
  colnames(OC[[scenario]])=sapply(1:ncol(OC[[scenario]]),FUN=function(x){paste0("cancer ",x)})
  rownames(OC[[scenario]])=sapply(1:nrow(OC[[scenario]]),FUN=function(x){paste0(methods[ceiling(x/2)]," - % ",ifelse(x/2!=floor(x/2),"reject","stop"))})
  OC[[scenario]][paste0(methods[5]," - % ","reject"),]=sapply(1:K,FUN=function(x){sum(decision.independent[,x]==1)/num.sim*100})
  OC[[scenario]][paste0(methods[5]," - % ","stop"),]=sapply(1:K,FUN=function(x){sum(decision.independent[,x]==-1)/num.sim*100})
  samplesize[scenario,]=samplesize[scenario,]/num.sim
  OC[[scenario]]=round(OC[[scenario]],1)
  Decision[[scenario]]=round(Decision[[scenario]],3)
  samplesize=round(samplesize,1)
  
  Bias[[scenario]]=matrix(NA,length(methods),K)
  colnames(Bias[[scenario]])=sapply(1:ncol(Bias[[scenario]]),FUN=function(x){paste0("cancer ",x)})
  rownames(Bias[[scenario]])=methods
  MSE[[scenario]]=matrix(NA,length(methods),K)
  colnames(MSE[[scenario]])=sapply(1:ncol(MSE[[scenario]]),FUN=function(x){paste0("cancer ",x)})
  rownames(MSE[[scenario]])=methods
  Bias[[scenario]][methods[5],]=colMeans(Bias.independent)
  Bias[[scenario]]=round(Bias[[scenario]],4)
  MSE[[scenario]][methods[5],]=Bias[[scenario]][methods[5],]^2 + apply(pest.independent,2,var)
  MSE[[scenario]]=round(MSE[[scenario]],4)
}