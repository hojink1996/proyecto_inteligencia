## Proyecto Imágenes (EL5206-1 Laboratorio de Inteligencia Computacional)
### Integrantes: Hojin Kang y Eduardo Salazar

#### Requisitos

<pre>
Para la prueba del proyecto se tienen una serie de requisitos que se indican a continuación.

    - Python versión 3.7
    - Librerías OpenCV2, matplotlib, pandas, numpy, sklearn, scipy y pickle
    - Descargar las carpetas con los videos y videos de ataque de la librería para poder correr
    el programa correctamente.
</pre>

#### Herramientas de la librería

<pre>
En las herramientas de la librería se debe cambiar los __init__ de las clases VideoOperator
y MultiVideo en el archivo <b>VideoOperator.py</b>. En los __init__ se debe cambiar el parámetro
<b>root_path</b> al directorio en el cual se guarde el proyecto, es decir en la carpeta en la 
cual se encuentre contenido todo el proyecto.
</pre>

#### Scripts de prueba

<pre>
Para correr el script de prueba se debe ir al archivo <b>values_testing.py</b>. En este archivo
se pueden cambiar los valores de los parámetros <i>user</i>, el cual es un número del 1 al 6 que
indica el usuario que se va a usar, <i>example_number</i> que es un número del 1 al 10 que corresponde
al ejemplo del usuario que se va a probar, y <i>video_type</i>, que puede ser 'ataque' o 'original'
e indica si el ejemplo que se va a probar es de una cara falsa o una real, respectivamente.

Si el método detecta que la cara es falsa, las imágenes saldrán con un cuadro rojo alrededor,
mientras que si detecta que es una cara real, saldrán con un cuadro azul alrededor. Es importante
notar que para pasar a la próxima imagen basta con apretar cualquier teclar.
</pre>

#### Reentrenar el SVM

<pre>
Si se quiere probar con un SVM con distintos parámetros, se puede ir al archivo <b>train_svm.py</b>.
En particular en la línea 166 del código se pueden cambiar los parámetros del SVM. Una vez se tienen
los parámetros y se corre el script, basta con volver a correr el script de prueba en <b>values_testing.py</b>
para probar el nuevo modelo.
</pre>
