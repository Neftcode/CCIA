/*
 * Autor: Carlos Arévalo
 * Fecha: 12 de marzo de 2022
 * Materia HPC-3
 * Tema: Operación de matrices con Eigen
 * Parcial I
*/

#include "matrixoperations.h"
#include <eigen3/Eigen/Dense>

Eigen::Matrix3d MatrixOperations::sumaMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2) {
    return m1+m2;
}

Eigen::Matrix3d MatrixOperations::restaMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2) {
    return m1-m2;
}

Eigen::Matrix3d MatrixOperations::productoMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2) {
    return m1-m2;
}

Eigen::Matrix3d MatrixOperations::matrizTranspuesta(Eigen::Matrix3d m) {
    return m.transpose();
}

Eigen::Matrix3d MatrixOperations::matrizInversa(Eigen::Matrix3d m) {
    return m.inverse();
}
