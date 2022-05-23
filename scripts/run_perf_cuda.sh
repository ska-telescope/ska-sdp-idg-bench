for i in {1..8}
do
    ./install/bin/cuda-gridder_v${i}
done

for i in {1..6}
do
    ./install/bin/cuda-degridder_v{i}
done
