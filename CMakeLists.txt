cmake_minimum_required(VERSION 3.2)
SET(CMAKE_CXX_STANDARD 14)
PROJECT(ree_net)

#find_package(Torch REQUIRED)
find_package(Torch
            PATHS lib/libtorch
            )

LINK_DIRECTORIES(lib/cfasttext/build/)
include_directories(lib/cfasttext/include)

aux_source_directory(src SRC)
aux_source_directory(lib/cfasttext/build/fasttext FASTTEXT)

add_executable(ree_net  examples/main.cc ${SRC} ${FASTTEXT})

target_link_libraries(ree_net libcfasttext.so libpthread.so "${TORCH_LIBRARIES}")
#set_property(TARGET ree_net PROPERTY CXX_STANDARD 14)


