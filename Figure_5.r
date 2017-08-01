# Figure 5: Comparing our measure to UK manifestos for post-1945 period.
# NOTE: you will need an API key for the Manifesto project. See:
# https://manifestoproject.wzb.eu/information/documents/api

rm(list=ls())


#load package and grab the dataset, subset to UK
require(manifestoR)
mp_setapikey("foo/your_api_key.txt") # change path to your key here.
mpds <- mp_maindataset()
mdata <- subset(mpds, countryname == "United Kingdom")

#grab RILE estimates
wparty <- which(mdata$partyname%in%c("Labour Party","Conservative Party"))
party.name <- mdata$partyname[wparty]
wdate  <- mdata$edate[wparty]
wrile  <- rile(mdata)[wparty]

dframe <- data.frame(wdate, party.name, wrile)

#colors
cols <- as.character(dframe$party.name)
cols[dframe$party.name=="Labour Party"] <- "red"
cols[dframe$party.name=="Conservative Party"] <- "lightblue"

#pchs
pchs <- as.character(dframe$party.name)
pchs[dframe$party.name=="Labour Party"] <- 22
pchs[dframe$party.name=="Conservative Party"] <- 21
pchs <- as.numeric(pchs)


#create difference plot -- 'polarization'
polar <- c()
for(i in 1:length(unique(dframe$wdate))){
  sub <- subset(dframe, dframe$wdate==unique(dframe$wdate)[i])
  diffe <- abs(sub$wrile[sub$party.name=="Conservative Party"] - sub$wrile[sub$party.name=="Labour Party"])
  polar <- c(polar, diffe)
}  

x11()
#put everything on same plot
par(bg='cornsilk1')
plot(wdate, wrile, col="black" , bg=cols, pch=pchs,  ylim=c(min(wrile),max(polar) ), cex=2,
     xlab="year", ylab="RILE")
lines(unique(dframe$wdate), polar, type="l", lwd="2" )
#looks v similar to our plot, though 1945/50 looks a bit different.

#put on a lowess
low <- lowess(polar ~ 1:length(unique(dframe$wdate)), f=1/4)
lines(unique(dframe$wdate), low$y, lwd=3, lty=2, col="purple")

legend('topright', pch=c(21,22), pt.cex=2, col='black', pt.bg=c("lightblue","red"), legend=c("Conservative", "Labour"))


#check correlation between the tseries for polarization estimates
# (needs bit of work bec the dates are different)


#grab dates from estimation of polarization from machine
wd <- getwd() # assumes this is run from the replication folder, with data in a subdirectory.
data.dir <- paste0(wd, "/data/")

out<- read.csv(paste0(data.dir, "acc_j27_allmembers.csv"))
out$max.mean <- apply(out[,3:6], FUN=max, MARGIN=1)
month <- out$yrmth
est.dates <- as.Date(paste(month,"-01",sep=""))

#grab dates from manifesto ests
uwdate <- unique(dframe$wdate)

vec.dates <- c()

for(i in 1:length(uwdate)){
  
  vec.dates <- c(vec.dates, which(abs(est.dates-uwdate[i]) == min(abs(est.dates - uwdate[i]))) )
  
}

#now do the correlations (~.16)
corr  <- cor(out$max.mean[vec.dates], polar)

#NBthat this is much higher when we exclude first couple of points:
vec.dates2 <- vec.dates[3: length(vec.dates)]
corr2  <- cor(out$max.mean[vec.dates2], polar[3:length(polar)])

