cmake_minimum_required(VERSION 3.14)
project("dlib activations benchmark" LANGUAGES CXX)

set(CPACK_PACKAGE_NAME "dlib-activations-benchmark")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

# Use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Colored warnings
option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
if(${FORCE_COLORED_OUTPUT})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options (-fdiagnostics-color=always)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options (-fcolor-diagnostics)
    endif()
endif()

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

# Optimization flags
include(CheckCXXCompilerFlag)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Dependency management
include(FetchContent)

# dlib
set(DLIB_TAG master)
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        ${DLIB_TAG}
    SOURCE_DIR     ${PROJECT_SOURCE_DIR}/external/dlib
)

message("-- Fetching dlib ${DLIB_TAG}")
FetchContent_MakeAvailable(dlib)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE dlib::dlib)
target_compile_options(main PRIVATE -Wall -Wextra -pedantic -Wno-deprecated-copy)
install(TARGETS main DESTINATION bin)
