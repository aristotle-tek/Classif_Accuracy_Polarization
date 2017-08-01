# Replication code for Figure 1: Classification Accuracy (y-axis)
# for Different Levels of Separation (x-axis)
# at different levels of noise.

rm(list=ls())
library(ggplot2)

wd <- getwd() # Run this from the replication folder, with data in a subdirectory (or setwd())
data.dir <- paste0(wd, "/data/")
df <- read.csv(paste0(data.dir, "acc_sims.csv"))

df <- df[,c("separation", "mn_sag",'noisefrac')]
df$sep <- .5-df$separation 


# fix noise / limit labels for clarity:
d.3 <- df[df$noisefrac==0.5,]
d.3 <- d.3[,c("separation", "mn_sag")]
d.3$sep <- .5-d.3$separation 
dlab <- d.3[c(11,13,15,17,19),]
df <- df[df$separation>.39,]


df.sel <- df[is.element(df$noisefrac, c(.05, .1, .25, .5)),]
pdf("separation_v_acc.pdf",  width=8, height=6)
p <- ggplot(df.sel, aes(x=separation, y=mn_sag, linetype=factor(noisefrac)))
p + geom_line() + theme_bw() +  scale_x_continuous(breaks=dlab$separation, labels=dlab$sep )+ 
  xlab("Separation") + ylab("Mean Classification Accuracy, 10-fold X-Validation")+labs(linetype = "Fraction Noise")
dev.off()
