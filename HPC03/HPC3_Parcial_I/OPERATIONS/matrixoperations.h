#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H

/*
 * Autor: Carlos Arévalo
 * Fecha: 12 de marzo de 2022
 * Materia HPC-3
 * Tema: Operación de matrices con Eigen
 * Parcial I
*/

#include <eigen3/Eigen/Dense>

class MatrixOperations {
public:
    MatrixOperations(){}

    /*Prototipo de funciones de la clase*/
    Eigen::Matrix3d sumaMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2);
    Eigen::Matrix3d restaMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2);
    Eigen::Matrix3d productoMatrices(Eigen::Matrix3d m1, Eigen::Matrix3d m2);
    Eigen::Matrix3d matrizTranspuesta(Eigen::Matrix3d m);
    Eigen::Matrix3d matrizInversa(Eigen::Matrix3d m);
};

#endif // MATRIXOPERATIONS_H
