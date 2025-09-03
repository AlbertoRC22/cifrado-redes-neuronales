from train import entrenar
from eval import evaluar
from draw_graphs import draw_graph
import time as t

if __name__ == '__main__':


    n_mensajes = 10000
    bits = 32
    batch_size = 512
    adam_optimizer = 0.001
    muestras = 10
    
    step = 300
    total_epochs = 6000

    n_ejecuciones = 3
    nombre_figura_base = f"Figura 11 - {batch_size}"

    for i in range(0, n_ejecuciones):
        epochs = 300

        if (i == 0):
            adam_optimizer_act = adam_optimizer / (10 ** i)
            nombre_figura = nombre_figura_base + f" y {adam_optimizer_act}"
            
            res_file_name = f"Resultados_11_{batch_size}_{adam_optimizer_act}.txt"
            with open(res_file_name, "w") as f:
                f.write("-- RESULTADOS DEL EXPERIMENTO 1.1. --\n\n")
        elif (i == 1):
            adam_optimizer_act = adam_optimizer / (10 ** i)
            nombre_figura = nombre_figura_base + f" y {adam_optimizer_act}"
            
            res_file_name = f"Resultados_11_{batch_size}_{adam_optimizer_act}.txt"
            with open(res_file_name, "w") as f:
                f.write("-- RESULTADOS DEL EXPERIMENTO 1.1. --\n\n")
        else:
            adam_optimizer_act = adam_optimizer / (10 ** i)
            nombre_figura = nombre_figura_base + f" y {adam_optimizer_act}"
            
            res_file_name = f"Resultados_11_{batch_size}_{adam_optimizer_act}.txt"
            with open(res_file_name, "w") as f:
                f.write("-- RESULTADOS DEL EXPERIMENTO 1.1. --\n\n")
            

        diccionario_medidas = {
            "epochs": [],
            "training_times": [],
            "media_precision": [],
            "media_distancias": [],
            "reconstrucciones_perfectas": []
        }

        with open(res_file_name, "a") as f:
            f.write(f"Número de mensajes = {n_mensajes}\n")
            f.write(f"Número de bits = {bits}\n")
            f.write(f"Tamaño del batch = {batch_size}\n")
            f.write(f"Adam optimizer learning rate = {adam_optimizer_act}\n")
            f.write(f"Epochs totales = {total_epochs}\n")
            f.write(f"Epochs iniciales = {epochs}\n")
            f.write("\n")

        tiempo_inicial = t.time()
        while epochs <= total_epochs:
            
            training_time = entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer)
            res_list = evaluar(n_mensajes, bits, muestras, res_file_name, epochs)

            media_precision = res_list[0]
            media_distancias = res_list[1]
            reconstrucciones_perfectas = res_list[2]

            diccionario_medidas["epochs"].append(epochs)
            diccionario_medidas["training_times"].append(training_time)
            diccionario_medidas["media_precision"].append(media_precision)
            diccionario_medidas["media_distancias"].append(media_distancias)
            diccionario_medidas["reconstrucciones_perfectas"].append(reconstrucciones_perfectas)

            epochs += step

        tiempo_total = t.time() - tiempo_inicial
        with open(res_file_name, "a") as f:
            f.write(f"Tiempo total de ejecución (s): {tiempo_total}\n\n")
        
        draw_graph(diccionario_medidas, n_mensajes, nombre_figura)