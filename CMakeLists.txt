cmake_minimum_required(VERSION 3.2)
SET(CMAKE_CXX_STANDARD 14)
PROJECT(reenet)

#find_package(Torch REQUIRED)
find_package(Torch
            PATHS lib/libtorch
            )

LINK_DIRECTORIES(lib/cfasttext/build/)
include_directories(lib/cfasttext/include)
include_directories(lib/rapidcsv/src)

aux_source_directory(src SRC)
aux_source_directory(lib/cfasttext/build/fasttext FASTTEXT)

add_executable(exe_${PROJECT_NAME}  examples/main.cc ${SRC} ${FASTTEXT})
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SRC} ${FASTTEXT})

target_link_libraries(exe_${PROJECT_NAME} libcfasttext.so libpthread.so "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} libcfasttext.so libpthread.so "${TORCH_LIBRARIES}")
#set_property(TARGET ree_net PROPERTY CXX_STANDARD 14)


