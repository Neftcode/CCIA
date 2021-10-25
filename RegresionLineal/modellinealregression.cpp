#include "modellinealregression.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>


/* Se necesita entrenar el modelo, lo que implica minimizar la función de costo
   de esta forma se puede medir la función de hipótesis. La función de costo
   es la forma de penalizar al modelo por cometer un error. */

float ModelLinealRegression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta) {
    Eigen::MatrixXd diferencia = pow((X*theta-y).array(), 2);
    return (diferencia.sum()/(2*X.rows()));
}

/* Se necesita proveer al algoritmo una función para dar los valores iniciales de
   de theta, el cual cambiará iterativamente hasta que converga al valor mínimo de
   nuestra función de costo. Básicamente esto representa el gradiente descendiente,
   el cual es las derivadas parciales de la función. Las entradas para la función
   será X (Features), y (Targets), alpha (learning rate) y el número de iteraciones
   (número de veces que se actualizará theta hasta que la función converga. */

std::tuple<Eigen::VectorXd, std::vector<float>> ModelLinealRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteraciones) {
    /* Almacenamiento temporal de thetas */
    Eigen::MatrixXd temporal = theta;
    /* Necesitamos la cantidad de parámetros m (Features) */
    int parametros = theta.rows();
    /* Costo inicial: se actualizará con los nuevos  pesos. */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X, y, theta));
    /* Por cada iteración se calcula la función de error. Se actualiza theta y se
       calcula el nuevo valor de la función de costro para los nuevos valores de theta. */
    for(int i=0; i<iteraciones; i++) {
        Eigen::MatrixXd error = X*theta -y;
        for (int j=0; j<parametros; j++) {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j, 0) = theta(j, 0) - ((alpha/X.rows())*termino.sum());
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X, y, theta));
    }
    return std::make_tuple(theta, costo);
}

/* Se crea la métrica r2 */
float ModelLinealRegression::R2Cuadrado(Eigen::MatrixXd y, Eigen::MatrixXd y_hat) {
    auto numerador = pow((y-y_hat).array(),2).sum();
    auto denomidador = pow(y.array()-y.mean(), 2).sum();
    return 1-(numerador/denomidador);
}
