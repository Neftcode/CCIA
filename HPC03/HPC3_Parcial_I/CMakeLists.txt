cmake_minimum_required(VERSION 3.5)

project(HPC3_Parcial_I LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Variable de fuentes
set(project_sources
    main.cpp
    OPERATIONS/matrixoperations.cpp)

# Variable de Interfaces
set(project_headers
    OPERATIONS/matrixoperations.h)

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
