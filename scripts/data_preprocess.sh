#! /bin/bash

cd data/UCF-Crimes/raw
for i in 1 2 3 4
do
  unzip Anomaly-Videos-Part-$i.zip
done
unzip Normal_Videos_for_Event_Recognition.zip
unzip UCF_Crimes-Train-Test-Split.zip

cd ..
mkdir videos
for i in 1 2 3 4
do
  mv raw/Anomaly-Videos-Part-$i/* videos/
  rm -rf raw/Anomaly-Videos-Part-$i
done
mv raw/Normal_Videos_for_Event_Recognition videos/Normal_Videos_event
mv raw/Action_Regnition_splits .