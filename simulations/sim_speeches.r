# basic simulations: create left and right parties in 
# simulated parliament with various amounts of noise

# Notes:
# (1) First, we define the fraction of the data which is noise ("noise.frac")
# (2) Then with what remains, the "signal" is divided into aa and bb according to "separation":
# "Separation" \in(.5, 1) is one number to generate the spectrum of divided vs consensual, as follows:
# say noise = .2, so the signal =.8. Then:
# consensual - "separation=.5" => a=.4, b=.4
# divided - "separation=.9" => a=.9, b=.1

#make a very divided, low noise, parliament with e.g.
###parl <- make.parliament(300, .1, .9, 300, .9, .1, 100 ) #this implies zero noise words.

#make a cosensual parliament with little noise with e.g.
### make.parliament(300, .4, .5, 300, .5, .4, 100)

#make a very noisy parliament with e.g.
### make.parliament(300, .2, .3, 300, .3, .2, 100)

rm(list=ls())

#define language
right.language <- paste("right_word",1:50, sep="_" )
left.language <- paste("left_word", 1:50, sep="_")
noise <- paste("noise_word", 1:50, sep="_")


#think of as MP's speech as simply being a random draw w replacement 
# from this set of language
make.mp <- function(a=.3, b=.3, c=1-(a+b), total.words=100){
  mp.speech <- c(sample(right.language, size=a*total.words ,replace=T),
                 sample(left.language, size= b*total.words ,replace=T),
                 sample(noise, size= c*total.words ,replace=T))
  paste(mp.speech, collapse=" ") #puts everything out as one long string
}

#so, special case includes e.g. 'pure' Tory MP: make.mp(1,0,0)
#and e.g. bipartisan MP with no noise make.mp(.5, .5)

#can make a party from those MPs
make.party <- function(strength=50, aa=.8, bb=.1, cc=1-(aa+bb), total.words=100){
  party.out <- t(replicate(strength, make.mp(aa, bb, cc))) #each row is an MP, each column is a speech
  party.out  
}
#for example, could have a left party as make.party(300, a=.1, b=.8, total.words=100) -- each MP makes 100 speeches

options(stringsAsFactors=FALSE) #just make sure speeches aren't factors

#make parliament of two parties
make.parliament <- function(strength.left=300, a.left=.1, b.left=.8, strength.right=300, a.right=.8, b.right=.1, total.words=100 ){
  left.party <- as.character( make.party(strength= strength.left, aa=a.left, bb=b.left, total.words=total.words) )
  right.party <-as.character( make.party(strength= strength.right, aa=a.right, bb=b.right, total.words=total.words) )
  
  Y <- c( rep(0, strength.left), rep(1, strength.right))
  MPs <- c( paste("MP.left",1:strength.left, sep="."),  paste("MP.right",1:strength.right, sep=".")  )
  dframe <- data.frame(MP=MPs, speeech=c(left.party, right.party))
  #now convert everything to a tfidf dtm
  require(quanteda)
  parliament.idf <- as.matrix( tfidf(dfm(corpus(dframe[,2], ), removeNumbers=F) ) ) 
  rownames(parliament.idf) <- MPs
  parliament.idf.y <- cbind(Y, parliament.idf)
  parliament.idf.y
}


strength <- 300
size <- 100

# separation: from parity- a/b = 1 to extreme - a/b = very large or very small
for(separation in c(0.5, 0.4, 0.3, 0.2, 0.1)){
  print(paste("separation:", separation), sep=' ')
  for(noise.frac in c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)){ #seq(0,1, .1)){
    print(paste("noise fraction:", noise.frac), sep=' ')
    signal <- 1-noise.frac
    a.left= signal * separation
    a.right= signal * (1-separation)
    b.left = 1- a.left
    b.right = 1- a.right
    parl <- make.parliament(strength, a.left, b.left, strength, a.right, b.right, size)
    write.csv(parl, file=paste("/nas/tz/uk/sim/sims/sim", strength, size, separation*100, noise.frac*100, ".csv", sep='_'))
  }
}

#------------------------------
#
#------------------------------------
separation <- 0.4


for(noise.frac in seq(0,1, .001)){
  print(paste("noise fraction:", noise.frac), sep=' ')
  signal <- 1-noise.frac
  a.left= signal * separation
  a.right= signal * (1-separation)
  b.left = 1- a.left
  b.right = 1- a.right
  parl <- make.parliament(strength, a.left, b.left, strength, a.right, b.right, size)
  write.csv(parl, file=paste("/nas/tz/uk/sim/sims/sim_1k_", strength, size, separation*100, noise.frac*100, ".csv", sep='_'))
}
}
