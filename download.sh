
URL=http://images.cocodataset.org/annotations/annotations_trainval2014.zip
ZIP_FILE=./data/coco_trainval_annotation.zip
mkdir -p ./data
wget -N $URL $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE


