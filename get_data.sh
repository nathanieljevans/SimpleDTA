#!/bin/sh
#######################################################################################################################
#######################################################################################################################


# pFam ; protein families
pfam=http://ftp.ebi.ac.uk/pub/databases/Pfam/Pfam-N/pfamA.tsv

#######################################################################################################################
#######################################################################################################################


# check that dir/file exists and then make/download
ROOT=./data
#[ ! -d "{$ROOT}" ] && echo 'deleting $ROOT' && rm -r $ROOT
[ ! -d "{$ROOT}" ] && mkdir $ROOT

#######################################################################################################################
#######################################################################################################################

# date of download
[ -f "$ROOT/date_of_download.txt" ] && rm $ROOT/date_of_download.txt
date > $ROOT/date_of_download.txt

#######################################################################################################################
#######################################################################################################################


[ ! -f "$ROOT/pfamA.tsv" ] && wget $pfam -O $ROOT/pfamA.tsv