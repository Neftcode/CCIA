/*
 * Autor: Carlos Arévalo
 * Fecha: 1 de marzo de 2022
 * Materia: HPC-3
 * Objetivo: Se requiere crear una aplicación
 * para el calculo de la regresión lineal.
 * 1. Debe haber una clase para el tratamiento,
 * manipulación normalización de los datos, dado
 * por un fichero CSV (valres separados por comas)
 * 2. Crear/Implementar una clase que haga los
 * calculos sobre el modelo de regresión
 * lineal, usando el gradiente descendiente
 *
*/

#include "EXTRACT/extracciondata.h"
#include "regresionlineal.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <boost/algorithm/string.hpp>

int main(int argc, char* argv[]){

    /* Se instancia el objeto del tipo "extracciondata"
     * y se incluyen los 3 argumentos que hemos de
     * pasar al objeto (dato por el constructor de la clase) */
    ExtractionData extraction(argv[1], argv[2], argv[3]);

    /* Se instancia el objeto Regresión Lineal a RL */
    RegresionLineal RL;

    /* Se leen los datos del fichero, a través
     * de la función ReadCSV() */
    std::vector<std::vector<std::string>> dataSet = extraction.Readcsv();

    /* Para probar la lectura del fichero,
     * se obtienen la cantidad de filas y columnas */
    int rows = dataSet.size() + 1;
    int cols = dataSet[0].size();

    /* Se crea una matriz dinámica double
     * de dimensión rows*cols */
    Eigen::MatrixXd df = extraction.CSVtoEigen(dataSet, rows, cols);

    // std::cout << "Promedio:\n" << extraction.Promedio(df) << std::endl;
    // std::cout << "\nDesviación estándar:\n" << extraction.DesvStandard(df.rowwise() - extraction.Promedio(df)) << std::endl;

    Eigen::MatrixXd dfNormal = extraction.Normalizador(df);
    // std::cout << "Normalización de datos:\n" << dfNormal << std::endl;

    /* Imprimir la función Normalizador (a ser verificado) */
    // std::cout << "\nNormalización:\n" << dfNormal;

    /* A continuación, se hará el primer módulo de Machine Learning:
     * Se requiere una clase de Regresión Lineal (Implementación e
     * Interfaz). Debe definir un constructor, importar las bibliotecas
     * necesarias. Se debe tener en cuenta que el método de Regresión
     * Lineal es un método estadístico que define la relación entre las
     * varibales independientes, con la variable dependiente. La idea
     * principal, es definir una LÍNEA RECTA (Hiperplano), con sus
     * correspondientes coeficientes (pendientes) y los puntos de
     * corte (y = 0).
     *
     * Se tienen diferentes métodos para resolver RL: se implementará
     * el método de los Mínimos Cuadrados Ordinarios (OLS). El OLS es
     * un método sencillo y computacionalmente económico. OLS presenta
     * una solución óptima para conjunto de datos complejos.
     *
     * Para el PRESENTE caso, se tiene un conjunto de datos (winedata.csv)
     * con múltiples variables independientes. Se necesita el algoritmo
     * llamado GRADIENTE DESCENDIENTE. El objetivo del GD es minimizar
     * la "FUNCIÓN de COSTO" */

    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    /* Declaramos un objeto para recibir la tupla empaquetada */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> packet = extraction.TrainTestSplit(dfNormal, 0.8);
    /* Se necesita desempaquetar la tupla (matrices dinámicas double) en 4 grupos de datos */
    std::tie(X_train, y_train, X_test, y_test) = packet;
    /* Se imprime el total de filas, las filas para entrenamiento, las filas testeo en sus dos sabores */
    /*std::cout << "Matrix original: " << dfNormal.rows() << std::endl;
    std::cout << "X_train: " << X_train.rows() << std::endl;
    std::cout << "y_train: " << y_train.rows() << std::endl;
    std::cout << "X_test: " << X_test.rows() << std::endl;
    std::cout << "y_test: " << y_test.rows() << std::endl << std::endl;*/

    /* Se necesita imprimir la cantidad de columnas por sabor */
    /*std::cout << "Matrix original: " << dfNormal.cols() << std::endl;
    std::cout << "X_train: " << X_train.cols() << std::endl;
    std::cout << "y_train: " << y_train.cols() << std::endl;
    std::cout << "X_test: " << X_test.cols() << std::endl;
    std::cout << "y_test: " << y_test.cols() << std::endl;*/

    /* Se tiene en cuenta que la regresión lineal es un método estadístico.
     * La idea principal es crear un hiperplano con tantas dimensiones
     * como variables independientes tenga el dataset (pendientes/pesos y punto de corte).
     *
     * Se hace la prueba del modelo:
     * - Se crea un vector para prueba y para entrenamiento inicializado en "unos", que
     * corresponde a los features (variables independientes). */
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensiona las matrices para ser ubicadas en los vectores
     * anteriores. Similar a reshape() de numpy. */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta, para pasar al algoritmo del GD,
     * básicamente es un vector de ceros del mismo tamaño de
     * entrenamiento. Adicional, se declara alfa y el número de
     * iteraciones. */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iterations = 1000;

    /* Se definen las variables de salida que representan los
     * coeficientes del vector de costo. */
    Eigen::VectorXd thetaOut; // coeficientes
    std::vector<float> costo;

    /* Se desempaqueta la tupla obtenida del objeto de la clase
     * Regresión Lineal */
    std::tuple<Eigen::VectorXd, std::vector<float>> salidaGD = RL.descentGradient(X_train, y_train, theta, alpha, iterations);
    std::tie(thetaOut_train, costo_train) = salidaGD;

    // std::cout << thetaOut << std::endl;

    /* Se quiere observar como decrece la función de costo */
    /*for (auto v : costo) {
        std::cout << v << std::endl;
    }*/

    /* A continuación, con propósitos de seguridad y trazabilidad
     * se exporta el vector de costo y el vector theta a ficheros */
    extraction.vectorToFile(costo, "vectorCosto.txt");
    extraction.matrixToFile(thetaOut, "vectorTheta.txt");

    /* Con el propósito de ajustar el modelo y hacer las predicciones
     * necesarioas, calculamos de nuevo el promedio y las desviaciones
     * estándar basadas en los datos para calcular y_hat (predicción
     * de los valores y, según el modelo).*/
    auto muPromedio = extraction.Promedio(df);
    auto muFeatures = muPromedio(0, 8);
    auto escaladaData = df.rowwise()-df.colwise().mean(); // cada fila del dataframe
    auto muEstandar = extraction.DesvStandard(escaladaData);
    auto devFeatures = muEstandar(0, 8);
    Eigen::MatrixXd y_train_hat = (X_train*thetaOut*devFeatures).array()+muFeatures;
    Eigen::MatrixXd y = df.col(8).topRows(13600);

    /* A continuación se determina que tan bueno es nuestro
     * modelo utilizando la métrica R2 */
    float metricaR2 = RL.R2Score(y, y_train_hat);
    std::cout << metricaR2 << std::endl;

    extraction.matrixToFile(y_train_hat, "prediccion.txt");
    // Las variables independientes presentan una relción sobre la varible dependiente apróx. 37%

    return EXIT_SUCCESS;
}
