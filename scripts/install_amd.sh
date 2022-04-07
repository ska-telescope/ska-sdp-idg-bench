rm -rf $INSTALL_DIR/../build
rm -rf $INSTALL_DIR/
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DGPU_BRAND=AMD -DBUILD_HIP=ON 
make -j
make install