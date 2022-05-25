#ifndef EXTRACTIONDATA_H
#define EXTRACTIONDATA_H

/*
 * Autor: Carlos Arévalo
 * Fecha: 1 de marzo de 2022
 * Materia: HPC-3
 * Objetivo: Aplicación para el cálculo
 * del modelo de regresión lineal.
 * Funcionalidad:
 * 1. Crear/Implementar una Interfaz para
 * la clase que manipula, extrae, explora
 * los datos, dado por un fichero CSV
 * (Valores separados por comas).
 * Argumentos de entrada:
 *          Nombre del fichero,
 *          Delimitador
 *          Cabecera
 *
*/

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <fstream>

/*
 * Constructor de la clase para
 * los argumentos de entrada
 */

class ExtractionData{
    /*Nombre del fichero*/
    std::string setDatos;
    /*Delimitador de CSV*/
    std::string delimitador;
    /*Si tiene cabecera o no el CSV*/
    bool header;

public:
    ExtractionData(std::string datos,
                   std::string separador,
                   bool head):
    setDatos(datos),
    delimitador(separador),
    header(head){}

    /*Prototipo de funciones de la clase*/
    std::vector<std::vector<std::string>> Readcsv();
    Eigen::MatrixXd CSVtoEigen(
            std::vector<std::vector<std::string>> setDatos,
            int filas, int columnas);
    auto Promedio(Eigen::MatrixXd datos) ->
    decltype(datos.colwise().mean());
    auto DesvStandard(Eigen::MatrixXd datos) ->
    decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain);
    void vectorToFile(std::vector<float> vectorData, std::string nameFile);
    void matrixToFile(Eigen::MatrixXd DataMatrix, std::string nameFile);
};
#endif // EXTRACTIONDATA_H
