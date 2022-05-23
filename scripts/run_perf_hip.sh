for i in {1..8}
do
    ./install/bin/hip-gridder_v${i}
done

for i in {1..6}
do
    ./install/bin/hip-degridder_v${i}
done
