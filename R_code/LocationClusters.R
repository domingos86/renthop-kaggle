library(sp)
library(rgdal)
library(geosphere)
library(jsonlite)
setwd("C:/Users/sandh/Dropbox/FRM/Bootcamp/Projects/Project 3/twosigma/train.json/")

train <- fromJSON("train1000.json")

lat <- unlist(unname(train$latitude))
long <- unlist(unname(train$longitude))
listing_id <- unlist(unname(train$listing_id))

# Covert the data into a R data frame
df <- data.frame(lat)
df['long'] <-long
df['listing_id'] <- listing_id

#Make sure all the points fall within the bounds of the coordinates
#defined below. If the coordinate falls outside, set it equal to a random point
#c(W 74?24'00"--W 72?51'00"/N 41?10'00"--N 40?28'00")
df$lat <- ifelse(df$lat > 41.1, sample(df$lat, 1),df$lat)
df$lat <- ifelse(df$lat < 40.28, sample(df$lat, 1),df$lat)

df$long <- ifelse(df$long < -74.24, sample(df$long, 1),df$long)
df$long <- ifelse(df$long > -72.51, sample(df$long, 1),df$long)


#convert the data frame into a Spatial Points dataframe
df.sp <- df
coordinates(df.sp)<- ~long+lat

# use the distm function to generate a geodesic distance matrix in meters
# this does not work with the full data set as R says that it cannot allocate
# a matrix of this size
mdist <- distm(df.sp)

# cluster all points using a hierarchical clustering approach
hc <- hclust(as.dist(mdist), method="complete")

# define the distance threshold, in this case 40 m to 1000m
m50 = 50
m100 = 100
m150 = 150
m200 = 200
m500 = 500
m1000 = 1000

# define clusters based on a tree "height" cutoff "m***" and add them to the SpDataFrame
df.sp$m50 <- cutree(hc, h = m50)
df.sp$m100 <- cutree(hc, h = m100)
df.sp$m150 <- cutree(hc, h = m150)
df.sp$m200 <- cutree(hc, h = m200)
df.sp$m500 <- cutree(hc, h = m500)
df.sp$m1000 <- cutree(hc, h = m1000)

trainLocationClusters <- as.data.frame(df.sp)

write.csv(trainLocationClusters, file = "C:/Users/sandh/Dropbox/FRM/Bootcamp/Projects/Project 3/twosigma/train.json/LocationClusters.csv")

