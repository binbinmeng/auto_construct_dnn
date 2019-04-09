cmake_minimum_required(VERSION 2.8)
project (test_onnx)

find_package(Protobuf)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.0")
include(FindCUDA)

find_package(CUDA REQUIRED)

if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN/pluginsRT)
    include_directories(${CMAKE_SOURCE_DIR}/TensorRT-5.0.0.10)
    include_directories(${CMAKE_SOURCE_DIR}/TensorRT-5.0.0.10/include)
    link_directories(${CMAKE_SOURCE_DIR}/TensorRT-5.0.0.10/lib)
    set(TRT_LIB_DIR ${CMAKE_SOURCE_DIR}/TensorRT-5.0.0.10/lib)

    set( HEADER
            include/tkDNN/kernels.h
            include/tkDNN/Layer.h
            include/tkDNN/Network.h
            include/tkDNN/NetworkRT.h
            include/tkDNN/pluginsRT
            include/tkDNN/tkdnn.h
            include/tkDNN/utils.h
            include/tkDNN/Yolo3Detection.h
            include/tkDNN/pluginsRT/ActivationLeakyRT.h
            include/tkDNN/pluginsRT/Int8Calibrator.h
            include/tkDNN/pluginsRT/RegionRT.h
            include/tkDNN/pluginsRT/ReorgRT.h
            include/tkDNN/pluginsRT/RouteRT.h
            include/tkDNN/pluginsRT/ShortcutRT.h
            include/tkDNN/pluginsRT/UpsampleRT.h
            include/tkDNN/pluginsRT/YoloRT.h
            TensorRT-5.0.0.10/include/NvInfer.h
            TensorRT-5.0.0.10/include/NvInferPlugin.h)

    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
    LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")

    add_executable(test_onnx test_dnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS} ${HEADER})
    target_link_libraries(test_onnx
            ${PROTOBUF_LIBRARIES}
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtkDNN.so
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${TRT_LIB_DIR}/libnvparsers.so
            ${TRT_LIB_DIR}/libnvinfer_plugin.so
            ${TRT_LIB_DIR}/libnvinfer.so
            /usr/local/cuda-9.0/lib64/libcudnn.so)

else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()