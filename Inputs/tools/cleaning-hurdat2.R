#' @author Trey McNeely
#' Parse the HURDAT2 dataset into an R-friendly data frame.

library(tidyverse)
library(stringr)

hurdat2 <- file('./pacific_raw.txt')
open(hurdat2, 'r')
raw <- readLines(hurdat2)
close(hurdat2, 'r')

storm.start <- grep('EP', raw)

storm.headers <- strsplit(raw[storm.start], ',') %>%
  lapply(trimws) %>%
  map2(storm.start, ., c)

get.storm <- function(header) {
  first.entry <- as.numeric(header[1]) + 1
  lines <- as.numeric(header[4])
  ID <- header[2]
  name <- header[3]
  
  storm <- read.table(text = raw[first.entry:(first.entry + lines - 1)],
                      sep = ',', stringsAsFactors = FALSE)[,1:8]
  colnames(storm) <- c('DATE', 'TIME', 'L', 'CLASS', 
                        'LAT', 'LONG', 'WIND', 'PRESS')
  storm <- cbind(ID = ID, NAME = name, storm)
  
  return(storm)
}

storms <- lapply(storm.headers, get.storm)
storms <- do.call('rbind', storms)

storms[storms$PRESS == -999,]$PRESS <- NA

storms$SEASON <- substr(storms$ID, 5, 8)
storms$NUM <- substr(storms$ID, 3, 4) %>% as.numeric

storms$DATE <- as.Date(paste(as.character(storms$DATE)), format = '%Y%m%d')
storms$TIME <- str_pad(storms$TIME, 4, pad = '0')

storms$TIMESTAMP <- as.POSIXct(paste(storms$DATE, storms$TIME), 
                               format = '%Y-%m-%d %H%M')

storms$NAME <- as.character(storms$NAME)

storms$LAT <- storms$LAT %>%
  gsub('N', '', .) %>%
  as.numeric

east <- grep("E", storms$LONG)
west <- grep("W", storms$LONG)

storms[east,]$LONG <- storms[east,]$LONG %>%
  gsub('E', '', .) %>%
  as.numeric

storms[west,]$LONG <- storms[west,]$LONG %>%
  gsub('W', '', .) %>%
  as.numeric %>%
  '*'(-1)

storms$LONG <- storms$LONG %>% as.numeric
