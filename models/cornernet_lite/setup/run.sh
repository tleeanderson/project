if [ ! -d "../repo" ]; then
  mkdir ../repo
fi
if [ ! -d "../repo/CornerNet-Lite" ]; then
  git clone git@github.com:princeton-vl/CornerNet-Lite.git ../repo/CornerNet-Lite
fi
if [ ! -d "../repo/coco" ]; then
  git clone git@github.com:cocodataset/cocoapi.git ../repo/coco
fi
