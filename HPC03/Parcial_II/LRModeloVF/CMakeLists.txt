cmake_minimum_required(VERSION 3.5)

project(LRModeloVF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Variable de fuentes
set(project_sources
    main.cpp
    regresionlineal.cpp
    EXTRACT/extracciondata.cpp
    )
#Variable de Interfaces
set(project_headers
    regresionlineal.h
    EXTRACT/extracciondata.h)

#Ejecutables
add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
