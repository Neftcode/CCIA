/*
 * Autor: Carlos Arévalo
 * Fecha: 1 de marzo de 2022
 * Materia: HPC-3
 * Objetivo: Aplicación para el calculo
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

#include "extracciondata.h"
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <boost/algorithm/string.hpp>

/*Se implementa la primera función miembro: lectura
 *del fichero CSV. Para ello, disponemos de un vecor
 *de vectores del tipo string, en donde se itera
 *linea por linea y se almacena el vector
 *de vectores del tipo string, cada registro o fila
 * La función retornará un vector de vectores tipo string
*/

std::vector<std::vector<std::string>> ExtractionData::Readcsv(){
    /*Se abre el fichero .csv de solo lectura*/
    std::ifstream Fichero(setDatos);
    /*Se crea el vector de vectores tipo string
      a retornar : tendrá los datos del dataset */
    std::vector<std::vector<std::string>> datosString;
    /*Se itera a traves de cada linea. Se divide el
      contenido según el delimitador provisto por el
      constructor */
    std::string linea = ""; //Variable para almacenar cada linea del dataset

    while(getline(Fichero,linea)){
        /*Cada linea de almacena en vectorFila*/
        std::vector<std::string> vectorFila;
        /*cada vector se divide segun delimitador*/
        boost::algorithm::split(vectorFila,linea, boost::is_any_of(delimitador));
        /*Cada fila se ingresa al vector de vectores*/
        datosString.push_back(vectorFila);

    }

    /*Se cierra el fichero*/
    Fichero.close();
    /*Se retorna el vector de vectores*/
    return datosString;

}

/*Segunda función miembro: Almacenar el vector de vectores
 *del tipo string en una matrix. La idea es presentar
 *el conjunto de datos similar a un objeto pandas
 * (DataFrame)*/

Eigen::MatrixXd ExtractionData::CSVtoEigen(
        std::vector<std::vector<std::string>> setDatos,
        int filas, int columnas){
        /*Identificar si tiene o no cabecera*/
        if(header==true){
            filas = filas-1;
        }
        /*Se itera sobre las filas y columnas, para almacenar
         * en la matrix de dimensión filasxcolumnas.
         * Basicamente, se le almacenará strings del vector:
         * que luego se pasa a "float" para ser manipulados
         */
        Eigen::MatrixXd dfMatriz(columnas,filas);

        int i,j;
        for(i=0; i<filas;i++){
            for(j=0; j<columnas; j++){
                dfMatriz(j,i) = atof(setDatos[i][j].c_str()); //atof casteo a fotante
            }
        }
        /*Se transpone la matriz para ser retornada*/
        return dfMatriz.transpose();
}

/* Se requiere implementar una función que calcule
 * el promedio de los datos (xcolumnas). La función
 * debe ser verificada con Pddof=0ython usando cualquier
 * biblioteca (pandas, sklearn, seaborn...).
 * En C++, existe el tipo de dato "auto" -> "decltype".
 * En muchos casos, la herencia del tipo de dato no es
 * evidente. El tipo de dato "auto" -> "decltype"
 * especifica el tipo de variable (deduce en tiempo de
 * compilación) que va a heredar la función. Es decir,
 * en la función, si el tipo de retorno es "auto", se
 * evaluará mediante la expresión para la deducción
 * del tipo de dato a retornar. */
auto ExtractionData::Promedio(Eigen::MatrixXd datos) ->
decltype(datos.colwise().mean()) {
    return datos.colwise().mean();
}

/* Para implementar la función de Desviación Estándar
 * datos = xi - x.promedio()
 * En esta función */
auto ExtractionData::DesvStandard(Eigen::MatrixXd datos) ->
decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()) {
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}


/* Se implementa la función que calcule la normalización
 * de los datos. Lo anterior para regular la escala o magnitud,
 * de los datos. Por lo tanto, asegurar la precisión de los
 * modelos de Machine Learning. */
Eigen::MatrixXd ExtractionData::Normalizador(Eigen::MatrixXd datos) {
    Eigen::MatrixXd datosEsc = datos.rowwise() - Promedio(datos);
    Eigen::MatrixXd normMatrix = datosEsc.array().rowwise()/DesvStandard(datosEsc);
    return normMatrix;
}

/* A continuación, se implementa la función para hacer la
 * división de datos en dos grupos. El primer grupo es para
 * entrenamiento, por lo general se usa del 70% al 80% del
 * total de los datos. El segundo grupo de datos es para
 * pruebas. Se requiere crear una función que devuelva dos
 * grupos de datos, seleccionados de forma aleatoria. */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExtractionData::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain) {
    int rows = datos.rows();
    int rowsTrain = round(sizeTrain*rows);
    int rowsTest = rows - rowsTrain;
    /* Con Eigen se puede especificar un bloque de una matrix,
     * seleccionando las filas superiores para el conjunto de
     * entrenamiento, y las demás para el conjunto de pruebas. */
    Eigen::MatrixXd trainMatrix = datos.topRows(rowsTrain);
    /* Una vez seleccionadas las filas superiores para entrena-
     * miento, se seleccionan las columnas a la izquierda
     * (OJO/WARNING: para este conjunto de datos) correspondiente
     * a las "features" o variables independientes.
     * Entonces se selecciona la cantidad de columnas -1 */
    Eigen::MatrixXd X_train = trainMatrix.leftCols(datos.cols()-1);
    /* Se selecciona la variable dependiente, en los datos de
     * entrenamiento. */
    Eigen::MatrixXd y_train = trainMatrix.rightCols(1);

    /* Se realiza el mismo procedimiento para el conjunto
     * de datos de Prueba, recordando que se tiene los datos
     * de la parte inferior de la matriz de entrada. La función
     * bottomRows devuelve la parte inferior de la matriz. */

    Eigen::MatrixXd testMatrix = datos.bottomRows(rowsTest);
    /* Una vez seleccionadas las filas inferiores para entrena-
     * miento, se seleccionan las columnas a la izquierda
     * (OJO/WARNING: para este conjunto de datos) correspondiente
     * al "Target" o variables independientes.
     * Entonces se selecciona la cantidad de columnas -1 */
    Eigen::MatrixXd X_test = testMatrix.leftCols(datos.cols()-1);
    /* Se selecciona la variable dependiente, en los datos de
     * entrenamiento. */
    Eigen::MatrixXd y_test = testMatrix.rightCols(1);

    /* Finalmente se retorna la tupla, que contiene los subconjuntos
     * de prueba y de entrenamiento.
     * Atención con la tupla enviada, dado que al ser usada es
     * necesario desempaquetarla. */
    return std::make_tuple(X_train, y_train, X_test, y_test);

}

/* A continuación, se crea una función para exportar los valores
 * de vector a archivo */
void ExtractionData::vectorToFile(std::vector<float> vectorData, std::string nameFile) {
    /* Se crea la salida de flujo de datos del fichero de entrada */
    std::ofstream salidaData(nameFile);
    /* Se escribe cada objeto del tipo float sobre dataVector, condicionado
     * por un cambio de línea */
    std::ostream_iterator<float> dataVector(salidaData, "\n");
    /* Se hace una copia de los objetos escritos sobre el vector data */
    std::copy(vectorData.begin(), vectorData.end(), dataVector);
}

/*Se crea la función para para exportar una matrix dinamica doble de tipo Eigen a un fichero.
 * Las exportaciones  a ficheros son criticas o significativas en tanto se tiene seguridad y control.
 * sobre lo resultados parciales obtenidos.
*/
void ExtractionData::matrixToFile(Eigen::MatrixXd DataMatrix, std::string nameFile){
    //Se crea la salida de datos stream o flujo de datos del ficehro de entrada
    std::ofstream salidaData(nameFile);
    //Si el fichero esta abierto, y no ha llegado al final, copie los datos de la matriz
    if(salidaData.is_open()){
        salidaData << DataMatrix << '\n';
    }
}

/* WARNING ******* ADVERTENCIA *******
 * se ha de estudiar los datos para saber las posiciones
 * sobre las columnas: Variables dependientes / variables
 * independientes */
