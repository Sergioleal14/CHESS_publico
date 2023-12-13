
# INSTRUCCIONES USO DE CÓDIGO PARALELIZADO

Si se requiere de las instrucciones para ejecutar e utilizar el módulo, estas se pueden ver sobre el Readme de la rama
antigua. 

A continuación se adjunta el contenido y epliación de las celdas de código necesarias para poder clonar y utilizar el repositorio completo. Dicho repositorio consta de una rama old en la que se encuentra el código antiguo, sin paralelizar mediante cuda y de una rama main en la cual se encuentra el código nuevo. 

Se recomienda utilizar un entorno de ejcución como Google Colab, en el que disponer de una GPU para que el funcionamiento sea correcto. 

Los siguientes comandos clonan sobre el directorio _/content_, ya existente en todas las versiones de Colab, el repositorio principal. Además se crea una carpeta dentro del repositorio llamada _/old_ en la que se puede encontrar el código de la version anterior.

%cd /content
!rm -rf ./*
!git clone -b old --single-branch https://github.com/Sergioleal14/CHESS_publico

En primer lugar, para la ejecución de la versión sobre la que se puede juagr, dentro del código antiguo, es necesario instalar una terminal auxiliar sobre Colab ya que este no dispone de ella y se requiere de una interfaz de usuario y de un _stdin_ para introducir los movimientos (toda dicha sintaxis se puede encontrar en el Readme.md de la carpeta _/old_). 

!pip install colab-xterm
%load_ext colabxterm
%xterm

Una vez hecho esto, se puede navegar desde la terminal para realizar la ejecución de el ejecutable _chess_ desde _/old_
