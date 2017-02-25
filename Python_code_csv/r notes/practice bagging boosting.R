df = read.csv('features_split.csv')
View(df)
#Bagging
#selecting the rows randomly and then averaging th eprediction

# bOsting
#selecting randomly the subset of columns to give us different subsets
#subset size by the rule of thumb is m = sqrt(p)

library(tree)
library(ISLR)
data = iris

help(Carseats)
attach(Carseats)
  
#xgboost will teach us when do you stop randomly cutting your model 
# mostly when your model does not imporve any further

#bagging and boosting are mostly always done together

#boosting algorithem

#f(x) =0 
#r =y
#b =1,23,4..B

y = f(x)
#fuction or output = f(x) + punishment in this case lambda(f(x))

#then we update residual or length of error  == ri <- ri - lambda(f(x))

#r2 = r1 -lambda(f2)
# = r0 - lambdaf1 - lambdaf2
# = y - (lambdaf1+lambdaf2)
#in the end we just add all the f(x) = sumation of lambda f(x) from b=1 to B
# depth of the tree?
# number of parameters to depth of the tree?
# boosting machine is slow lerner so we will have to itterate through it again and again many times n.trees>>

#Boosting tunning parameters

#ususally booosting is better than random forest and also much faster

# VARIABLE IMPORTANCE IS DECIDED BY THE DECREASE IN RESIDUAL A PREDICTOR CAUSES FROM TOTAL REDUCTION

