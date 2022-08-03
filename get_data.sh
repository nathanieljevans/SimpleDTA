#!/bin/sh
#######################################################################################################################
#######################################################################################################################



# CLUE compound data
# info: https://clue.io/releases/data-dashboard
l1000_phaseII_compoundinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt

# gene info 
l1000_phaseII_geneinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
 
# Targetome 
targetome=https://raw.githubusercontent.com/ablucher/The-Cancer-Targetome/master/results_070617/Targetome_FullEvidence_070617.txt


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

[ ! -f "$ROOT/compoundinfo_beta.txt" ] && wget $l1000_phaseII_compoundinfo -O $ROOT/compoundinfo_beta.txt
[ ! -f "$ROOT/geneinfo_beta.txt" ] && wget $l1000_phaseII_geneinfo -O $ROOT/geneinfo_beta.txt
[ ! -f "$ROOT/targetome.txt" ] && wget $targetome -O $ROOT/targetome.txt

