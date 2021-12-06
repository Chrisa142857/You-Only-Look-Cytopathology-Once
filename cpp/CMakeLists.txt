cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

file(GLOB SOURCES
    ./*.h
    ./*.cpp
)
add_executable(main ${SOURCES})
add_library(tile_slide tile_slide.cpp)

find_package(Torch REQUIRED)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(tile_slide "${TORCH_LIBRARIES}")

find_package(PkgConfig)
pkg_check_modules(OPENSLIDE REQUIRED openslide)
include_directories(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CFLAGS
	${OPENSLIDE_CFLAGS} ${OPENSLIDE_CFLAGS_OTHER}
)
add_definitions(${CFLAGS})
target_link_libraries(main ${OPENSLIDE_LIBRARIES})
target_link_libraries(tile_slide ${OPENSLIDE_LIBRARIES})

install(TARGETS tile_slide DESTINATION lib)
install(FILES tile_slide.h DESTINATION include)

set_property(TARGET main PROPERTY CXX_STANDARD 14)