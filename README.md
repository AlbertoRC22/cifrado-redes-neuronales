Este es un proyecto que ilustra diferentes escenarios posibles en el campo de la criptografía utilizando redes neuronales, separando varios proyectos en diferentes directorios:

1. Alice y Bob

En este caso se modela el caso de dos agentes honestos: Alice y Bob. Alice es el agente que cifra los mensajes para que Bob los descifre.
Ilustra el caso en el que entrenan juntos y no hay atacante, con dos opciones posibles: cifrar con o sin clave.

1.5. Alice y Bob por separado

En este caso se ilustra la difcultad que entraña cifrar sin juntar los modelos de redes neuronales. Aún utilizando una sola clave, como Alice y Bob no aparenden a cifrar y descifrar el uno para el otro, Bob no consigue tener una precisión mayor al 50%.

2. Alice y Bob con atacante

Modela dos situaciones: El atacante, en este caso es Eve, entrena con Alice y Bob o ataca por su cuenta, obteniendo precisiones diferentes en el descifrado, mientras que Bob descifra bien los mensajes en ambos escenarios.
   
3. Alice, Bob, Charlie e Eve

En este caso, se ha añadido un agente honesto más e Eve, sin éxito, intenta descifrar los mensajes estando en el entrenamiento con ellos. Con un buen uso de las funciones de pérdida, podemos conseguir que el atacante no pueda romper el cifrado de las partes honestas.
