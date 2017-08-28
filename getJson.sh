FOL=jsoncpp
REPO=https://github.com/open-source-parsers/$FOL.git
git clone $REPO
cd $FOL
python amalgamate.py
cp -r dist/* ../src/utilities
cd ..
rm -rf $FOL
