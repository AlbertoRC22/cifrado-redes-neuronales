from train import entrenar
from eval import evaluar
from draw_graphs import draw_graph
import time as t
import math

if __name__ == '__main__':


    n_mensajes = 10000
    bits = 32
    batch_size = 128
    adam_optimizer = 0.001
    muestras = 10
    
    step = 300
    total_epochs = 6000

    nombre_figura_base = f"Figura 12 - Batch size = "

    batch_size_act = batch_size
    batch_size_limit = 512

    while batch_size_act <= batch_size_limit:
        epochs = 300

        nombre_figura = nombre_figura_base + f"{batch_size_act}"
        res_file_name = f"Resultados_12_{batch_size_act}.txt"
        with open(res_file_name, "w") as f:
            f.write("-- RESULTADOS DEL EXPERIMENTO 1.2. --\n\n")
            

        diccionario_medidas = {
            "epochs": [],
            "training_times": [],
            "media_precision": [],
            "media_distancias": [],
            "reconstrucciones_perfectas": [],
            "precision_errores": [],
            "dist_errores": [],
            "reconstrucciones_perfectas_errores": []
        }

        with open(res_file_name, "a") as f:
            f.write(f"Número de mensajes = {n_mensajes}\n")
            f.write(f"Número de bits = {bits}\n")
            f.write(f"Tamaño del batch = {batch_size_act}\n")
            f.write(f"Adam optimizer learning rate = {adam_optimizer}\n")
            f.write(f"Epochs totales = {total_epochs}\n")
            f.write(f"Epochs iniciales = {epochs}\n")
            f.write("\n")

        tiempo_inicial = t.time()
        while epochs <= total_epochs:
            
            training_time = entrenar(n_mensajes, bits, epochs, batch_size_act, adam_optimizer)
            res_list = evaluar(n_mensajes, bits, muestras, res_file_name, epochs)

            media_precision = res_list[0]
            media_distancias = res_list[1]
            reconstrucciones_perfectas = res_list[2]
            media_errores = res_list[3]
            dist_errores = res_list[4]
            reconstrucciones_perfectas_errores = res_list[5]

            diccionario_medidas["epochs"].append(epochs)
            diccionario_medidas["training_times"].append(training_time)
            diccionario_medidas["media_precision"].append(media_precision)
            diccionario_medidas["media_distancias"].append(media_distancias)
            diccionario_medidas["reconstrucciones_perfectas"].append(reconstrucciones_perfectas)
            diccionario_medidas["precision_errores"].append(media_errores)
            diccionario_medidas["dist_errores"].append(dist_errores)
            diccionario_medidas["reconstrucciones_perfectas_errores"].append(reconstrucciones_perfectas_errores)

            epochs += step
        
        exp = math.log(batch_size_act, 2)
        exp_entero = int(exp)
        batch_size_act = 2 ** (exp_entero + 1)

        tiempo_total = t.time() - tiempo_inicial
        with open(res_file_name, "a") as f:
            f.write(f"Tiempo total de ejecución (s): {tiempo_total}\n\n")
        
        draw_graph(diccionario_medidas, n_mensajes, nombre_figura)