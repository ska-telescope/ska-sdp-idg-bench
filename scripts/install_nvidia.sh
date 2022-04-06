rm -rf $INSTALL_DIR/../build
rm -rf $INSTALL_DIR/
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DGPU_BRAND=NVIDIA -DBUILD_CUDA=ON -DBUILD_HIP=OFF
make -j
make install
