#mkdir release
#g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native
g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -ggdb -march=native
