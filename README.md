Este es un proyecto que ilustra diferentes escenarios posibles en el campo de la criptografía utilizando redes neuronales, separando varios proyectos en diferentes directorios:

## 1. Alice y Bob

En este caso se modela el caso de dos agentes honestos: Alice y Bob. Alice es el agente que cifra los mensajes para que Bob los descifre.

Hay tres casos:

* el 1.1 trata de Alice y Bob cifrando y descifrando sin clave.
* el 1.2 trata de Alice y Bob cifrando y descifrando con clave.
* el 1.3 trata de Alice y Bob cifrando y descifrando, pero entrenados de forma separada.

## 2. Alice y Bob con atacante

En este caso se modelan dos escenarios de Alice y Bob con un atacante:
 
* 2.1 Alice Y Bob entrenan junto a Eve
* 2.2 Alice y Bob no entrenan junto a Eve

Los ficheros main.py de cada proyecto representan la útlima ejecución realizada, pero se hicieron más. Para no saturar el repositorio con ficheros main, lo que se ha hecho es inlcuir los resultados tanto en formato textual como las gráficas generadas en cada caso en el directorio resultados_finales/. Así, solo se ha incluido un fichero main.py por cada uno, ya que los demás son variaciones de este con diferentes parámetros o bucles para cambiar el número de ejecuciones, pero la lógica central es la misma en todos los main.py ejecutados.

El directorio resultados_finales/ tiene todas las gráficas y resultados textuales generados en cada escenario. En los nombres de los ficheros se incluye información acerca de los parámetros usados, pero en los ficheros .txt asociados se incluye al principio una sección con el valor concreto de cada parámetro.

El directorio jupyter/ contiene las Jupyter Notebooks de cada escenario. Aquí, por ser ejemplos ilustrativos con anotaciones, se ha incluido una versión básica del main en la sección del código principal que ejecuta el proyecto una sola vez con valores básicos. Por ejemplo, el escenario 1.1. tiene una versión que lo ejecutaría una vez con tamaño de _batch_ 512 y ratio de aprendizaje del optimziador Adam 0,001. Esto es para tener un caso básico y porque las notebooks permiten cambiar fácilmente los valores para probar otras versiones con valores diferentes.

--------------------------------------------------------------------------------------------------------------------------------------

Para ejecutar los proyectos, hay dos opciones: Ejecutar los ficheros main de cada uno o hacerlo con las Jupyter Notebooks.

## Dependencias

1. Python: Hay que descargarlo desde la página oficial, que es https://www.python.org/downloads/.
2. Keras y Tensorflow: Se instalan ejecutando "pip install tensrflow", ya que vienen integrados desde la versión 2.0.
3. Jupyter Notebooks: Se instala ejecutando el comando "pip install notebook".

Las versiones concretas utilizadas en el desarrollo de este proyecto han sido Python 3.11.9, TensorFlow 2.19.0, Keras 3.10.0 y Jupyter Notebook 7.4.5.

## Ejecución con los ficheros .py

En este caso, basta con utilizar el terminal para situarse en el subdirectorio correspondiente y ejecutar el comando "python ./main,py".

## Ejecución con las Jupyter Notebooks

Hay que dirigirse al directorio raíz o al directorio jupyter en una terminal para ejecutar ahí el comando "jupyter notebook", lo que abrirá una interfaz gráfica en la que se tendrá acceso a todas las notebooks y con la que se podrán tanto ejecutar como modificar.


