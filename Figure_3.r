# Figure 3: Estimates of parliamentary polarization, by session.
rm(list=ls())

wd <- getwd() # assumes this is run from the replication folder, with data in a subdirectory.
data.dir <- paste0(wd, "/data/")

out<- read.csv(paste0(data.dir, "acc_j27_allmembers.csv"))

dat <- read.csv(paste0(data.dir, "years_session_index.csv"))
#dat <- read.csv(paste0(data.dir, "years_d4.csv"))

#colors and pch
ps <-  as.character(dat$partypm)
cols<- ps
cols[ps=="con"] <- "lightblue"
cols[ps=="lab"] <- "red"
cols[ps=="war"] <- "gray"
cols[ps=="coalition"] <- "cadetblue2"

pchs <- ps
pchs[ps=="con"] <- 21
pchs[ps=="lab"] <- 22
pchs[ps=="war"] <- 23
pchs[ps=="coalition"] <- 24
pchs <- as.numeric(pchs)


out$max.mean <- apply(out[,3:6], FUN=max, MARGIN=1)
out$max.median <-  apply(out[,7:10], FUN=max, MARGIN=1)

require(strucchange)
bp <- breakpoints(out$max.mean~1)
elections <- which( dat$ge==1 )

x11()
par(bg='cornsilk1', las=2)
plot(1:length(out$max.mean), out$max.mean,  axes=F, type="b", lwd=2, xlab="election (year-month)", ylab="accuracy", pch=pchs, col='black', 
     bg=cols, cex=2)
abline(v=bp$breakpoints, lwd=2, col="darkgreen")


axis(1, at=c(1, bp$breakpoints, elections, nrow(out)  ), 
     labels=out$yrmth[c(1, bp$breakpoints, elections, nrow(out))], cex.axis=.7 )
axis(2)
box()

legend("topleft", pch=c(21, 22, 23, 24), col=c('black'), pt.bg=c('lightblue', 'red', 'gray', 'cadetblue2'), legend=c("Conservative", "Labour", "War", "Coalition") , 
       cex=1.5, bty="n")
