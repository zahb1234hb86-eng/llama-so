cmake_minimum_required(VERSION 3.22)
project(chat LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# llama.cpp كمكتبة ساكنة تُدمج داخل ملف واحد libchat.so
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # ARMv8: تعليمات dotprod تسرّع الضرب كثيراً (موجودة في Helio G88 وأغلب الأجهزة الحديثة)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod" CACHE STRING "" FORCE)
else()
    # ARMv7 (32 بت): إعدادات أبسط وأكثر توافقاً
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(llama
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5500.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(llama)

add_library(chat SHARED chat.cpp)
target_include_directories(chat PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include)
target_link_libraries(chat PRIVATE llama ggml android log)
target_link_options(chat PRIVATE -Wl,-z,max-page-size=16384)
