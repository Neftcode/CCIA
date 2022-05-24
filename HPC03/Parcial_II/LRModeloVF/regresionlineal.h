#ifndef REGRESIONLINEAL_H
#define REGRESIONLINEAL_H

/*
 * Autor: Carlos Arévalo
 * Fecha: 1 de marzo de 2022
 * Materia: HPC-3
 * Objetivo: Aplicación para el cálculo
 * del modelo de regresión lineal.
 *
*/

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

class RegresionLineal {
public:
    RegresionLineal(){}

    float OLS_costo(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> descentGradient(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iterator);
    float R2Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // REGRESIONLINEAL_H
