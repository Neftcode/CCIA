/*
 * Autor: Carlos Arévalo
 * Fecha: 12 de marzo de 2022
 * Materia HPC-3
 * Tema: Operación de matrices con Eigen
 * Parcial I
*/

#include "OPERATIONS/matrixoperations.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <ctime>   // Tiempo

/**
 * Retorna el tiempo en una cadena
 *
 * @return  string  Cadena de tiempo
 */
std::string getTime() {
    // current date/time based on current system
    time_t now = time(0);
    std::string dt = ctime(&now); // convert now to string form
    // convert now to tm struct for UTC
    // tm *gmtm = gmtime(&now);
    // dt = asctime(gmtm);
    dt.replace(dt.find("\n"), 1, ""); // quitar salto de línea a la cadena fecha
    return dt;
}

int main() {
    std::cout << "\n╔═════════════════════════════════════════════════════════════════╗\n"
             << "║                                                                 ║\n"
             << "║                       PARCIAL I DE HPC-03                       ║\n"
             << "║                                                                 ║\n"
             << "║                                                                 ║\n"
             << "║       Programa realiza operaciones de matrices con Eigen.       ║\n"
             << "║                                                                 ║\n"
             << "║                                                                 ║\n"
             << "║                   Universidad Sergio Arboleda                   ║\n"
             << "║                                                                 ║\n"
             << "║       Asignatura: HPC-3                                         ║\n"
             << "║       Tema: Operación de matrices con Eigen                     ║\n"
             << "║       Docente: John Jairo Corredor                              ║\n"
             << "║                                                                 ║\n"
             << "║       Autor: Carlos Alberto Arévalo Martínez                    ║\n"
             << "║       Fecha: " << getTime() << "                           ║\n"
             << "║                                                                 ║\n"
             << "╚═════════════════════════════════════════════════════════════════╝\n\n";


    // inicializar variables
    Eigen::Matrix3d matrixA, matrixB, matrixC;

    // crear objeto de la clase
    MatrixOperations operations;

    // asignar valores
    matrixA << 2, 0, 1, 3, 0, 0, 5, 1, 1;
    matrixB << 1, 0, 1, 1, 2, 1, 1, 1, 0;

    // suma de matrices
    matrixC = operations.sumaMatrices(matrixA, matrixB);
    std::cout << "Suma de matrices A + B:\n" << matrixC << "\n\n";

    // resta de matrices
    matrixC = operations.restaMatrices(matrixA, matrixB);
    std::cout << "Resta de matrices A - B:\n" << matrixC << "\n\n";

    // producto de matrices A*B
    matrixC = operations.productoMatrices(matrixA, matrixB);
    std::cout << "Producto de matrices A * B:\n" << matrixC << "\n\n";

    // producto de matrices B*A
    matrixC = operations.productoMatrices(matrixB, matrixA);
    std::cout << "Producto de matrices B * A:\n" << matrixC << "\n\n";

    // traspuesta de A
    std::cout << "Traspuesta de A:\n" << operations.matrizTranspuesta(matrixA) << "\n\n";

    // demostrar que A² - A - 2I = 0
    matrixA = matrixA.Ones();
    matrixA.diagonal() << 0, 0, 0;
    Eigen::MatrixPower<Eigen::Matrix3d> apow(matrixA);
    std::cout << "Demostrar que: A² - A - 2I = 0, siendo A=\n" << matrixA << "\n\n";
    matrixC = apow(2) - matrixA - 2*matrixB.Identity();
    std::cout << "A² - A -2I =\n" << matrixC << "\n\n";

    // n-ésima potencia de la matriz
    matrixA = matrixA.Identity();
    matrixA.row(0).col(2).array() = 1;
    int n;
    std::cout << "Ingrese n-ésima potencia para calcular la matriz: ";
    std::cin >> n;
    Eigen::MatrixPower<Eigen::Matrix3d> apow2(matrixA);
    std::cout << "\nn-ésima potencia de la matriz, siendo A=\n" << matrixA << "\n\n" << apow2(n) << "\n\n";

    // matriz inversa
    matrixA << 1, -1, 0, 0, 1, 0, 2, 0, 1;
    std::cout << "Matriz inversa de A, siendo A=\n" << matrixA << "\n\n" << operations.matrizInversa(matrixA) << "\n\n";

    return 0;
}
