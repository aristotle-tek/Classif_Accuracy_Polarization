# Figure 4: Mean Variance by Session

rm(list=ls())

library(ggplot2)

# setwd("~/Dropbox/machine_parliament/PA_replication/")
wd <- getwd() # Run this from the replication folder, with data in a subdirectory (or setwd())
data.dir <- paste0(wd, "/data/")

df<- read.csv(paste0(data.dir, "SAG_speaker_prob_estimates_allmembers_j27.csv"))

# Aggregate by index (session):
ag <- aggregate(df[,c('estprob','estprobvar')], by=list(df$index), FUN=mean)
names(ag) <- c('index', 'estprob','estprobvar')

# Session index -- 
dd <- read.csv(paste0(data.dir,'acc_j27_allmembers.csv'), stringsAsFactors=F)
yrs4 <- read.csv(paste0(data.dir,'years_session_index.csv'), stringsAsFactors=F)
dd$yrmth <- yrs4$yearmth


dd2 <- dd[seq(5,75, 5),] # subset for labeling the plot

a <- aggregate(df, by=list(df$index), FUN=mean)
ag2 <- ag[seq(5,75, 5),]
ag2$yrmth <- dd2$yrmth

p <- ggplot(ag, aes(x=index, y=estprobvar))
p + geom_line() + theme_bw()+ ylab('Mean variance of MP estimates' ) + xlab("Session of Parliament") + 
  scale_x_continuous(breaks=ag2$index, labels=ag2$yrmth)+ theme(axis.text.x = element_text(angle = 90, hjust = 1))# , angle = 90)

ggsave('plot_var_sess.pdf',  width=10, height=6)

p <- ggplot(ag, aes(x=year, y=estprobvar))
p + geom_line() + theme_bw()+ ylab('Mean variance of Speakers estimates' ) #+scale_x_continuous(breaks=dd2$index, labels=dd2$yrmth)+
