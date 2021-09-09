
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=\
#$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
#'https://docs.google.com/uc?export=download&id=19eq5JROsCWEMxrqMnS-O9evy5w4GBWyp' -O- |\
# sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19eq5JROsCWEMxrqMnS-O9evy5w4GBWyp" \
# -O COCO_VCOCO_annotation.zip && rm -rf /tmp/cookies.txt
#unzip COCO_VCOCO_annotation.zip
#rm COCO_VCOCO_annotation.zip
#rm __MACOSX*






COCO_TRAIN_IMG_URL=http://images.cocodataset.org/zips/train2014.zip
COCO_VAL_IMG_URL=http://images.cocodataset.org/zips/val2014.zip

COCO_IMG_DIR=./data/images
COCO_TRAIN_IMG_FILE="${COCO_IMG_DIR}/train2014.zip"
COCO_VAL_IMG_FIEL="${COCO_IMG_DIR}/val2014.zip"

#wget $COCO_TRAIN_IMG_URL -O "${COCO_IMG_DIR}/train2014.zip"
wget $COCO_VAL_IMG_URL -O "${COCO_IMG_DIR}/val2014.zip"

#unzip "${COCO_IMG_DIR}/train2014.zip" -d $COCO_IMG_DIR
unzip "${COCO_IMG_DIR}/val2014.zip" -d $COCO_IMG_DIR

#unzip $COCO_IMG_DIR_FILE





#
#COCO_ANNO_DIR=./data/coco/
#VCOCO_ANNO_DIR=./data/vcoco/annotations
#
#
#COCO_ANNO_ZIP_FILE="${COCO_ANNO_DIR}/coco_trainval_annotation.zip"
##COCO_TRAIN_IMG_FILE=
##COCO_VAL_IMG_FILE=
#
#mkdir -p $COCO_ANNO_DIR
#wget $COCO_ANNO_URL -O $COCO_ANNO_ZIP_FILE
#unzip $COCO_ANNO_ZIP_FILE -d $COCO_ANNO_DIR
#rm $COCO_ANNO_ZIP_FILE


#URL=http://images.cocodataset.org/zips/train2014.zip
#ZIP_FILE=./data/coco_train_image.zip
#mkdir -p ./data
#wget -N $URL $ZIP_FILE
#unzip $ZIP_FILE -d ./data/coco/images
#rm $ZIP_FILE
#

