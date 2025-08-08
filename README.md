Este es un proyecto que ilustra diferentes escenarios posibles en el campo de la criptografía utilizando redes neuronales, separando varios proyectos en diferentes directorios:

1. Alice y Bob

En este caso se modela el caso de dos agentes honestos: Alice y Bob. Alice es el agente que cifra los mensajes para que Bob los descifre.

Hay tres casos:

    - el 1.1 trata de Alice y Bob cifrando y descifrando sin clave.
    - el 1.2 trata de Alice y Bob cifrando y descifrando con clave.
    - el 1.3 trata de Alice y Bob cifrando y descifrando, pero entrenados de forma separada.

2. Alice y Bob con atacante

En este caso se modelan varios escenarios de Alice y Bob con un atacante:
 
    - 2.1 Alice Y Bob entrenan junto a Eve
    - 2.2 Alice y Bob no entrenan junto a Eve
    - 2.3 Alice y Bob entrenan junto a Eve, ademmás de Eve otra vez por separado, pero la función de pérdida ignora a Eve
