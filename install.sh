#!/bin/bash
set -e

mkdir -p data
cd data

echo "Downloading Indian Pines..."
wget -c http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat
wget -c http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat

echo "Downloading Pavia University..."
wget -c http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
wget -c http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat

echo "Downloading Salinas..."
wget -c http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat
wget -c http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat

echo ""
echo "Done. Files downloaded into ./data"
ls -lh
