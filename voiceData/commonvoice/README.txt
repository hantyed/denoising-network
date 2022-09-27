This data set is a subset of the Common Voice English language data set from Mozilla.
The version subset is en_1488h_2019-12-10.


The data was subset as follows:

Train - 
The contents of the train folder correspond to the train set enumerated by train.tsv in en_1488h_2019-12-10.
Subset to only files that have associated gender and age information.
Subset to not include files that have associated age marked as 'teens'.
Subset to only include files with three or more up_votes.
Subset to only include 1000 files marked as gender = 'male' and 1000 files marked as gender = 'female'.
The train.tsv file was similarly subset from the original train.tsv file to only reference the subset files.

Validation -
The contents of the validation folder correspond to the dev set enumerated by dev.tsv in en_1488h_2019-12-10.
Subset to only include files that have associated gender and age information.
Subset to not include files that have associated age marked as 'teens'.
Subset to only include files with two or more up_votes.
Subset to only include 200 files marked as gender = 'male' and 200 files marked as gender = 'female'.
The validation.tsv file was similarly subset (and renamed) from the original dev.tsv file to only reference the subset files.

Test -
The contents of the test folder correspond to the test set enumerated by test.tsv in en_1488h_2019-12-10.
Subset to only include files that have associated gender and age information.
Subset to not include files that have associated age marked as 'teens'.
Subset to only include 200 files marked as gender = 'male' and 200 files marked as gender = 'female'.
The test.tsv file was similarly subset from the original test.tsv file to only reference the subset files.


For all sets, the audio data files were converted from MP3 to WAV.