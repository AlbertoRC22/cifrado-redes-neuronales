from train import entrenar
from eval import evaluar
from draw_graphs import draw_graph
import time as t

if __name__ == '__main__':

    nombre_figura_base = "Figura 22 - 128 y 0,001 cada 500"
    res_file_name_base = "Resultados_22"

    n_mensajes = 10000 
    bits = 32
    step = 500
    total_epochs = 6000
    batch_size = 128
    adam_optimizer = 0.001
    beta = 2.0
    gamma = 7.0
    muestras = 10

    for i in range(0, 2):
    
        epochs = 500
        if (i == 0):
            res_file_name = res_file_name_base + "_2.txt"
            nombre_figura = nombre_figura_base + ", segunda vez"
        elif (i == 1):
            res_file_name = res_file_name_base + "_3.txt"
            nombre_figura = nombre_figura_base + ", tercera vez"

        with open(res_file_name, "w") as f:
            f.write("-- RESULTADOS DEL EXPERIMENTO 2.2. --\n\n")
    
        with open(res_file_name, "a") as f:
            f.write(f"Número de mensajes = {n_mensajes}\n")
            f.write(f"Número de bits = {bits}\n")
            f.write(f"Tamaño del batch = {batch_size}\n")
            f.write(f"Adam optimizer learning rate = {adam_optimizer}\n")
            f.write(f"Epochs totales = {total_epochs}\n")
            f.write(f"Epochs iniciales = {epochs}\n")
            f.write(f"beta = {beta}\n")
            f.write(f"gamma = {gamma}\n")
        
        diccionario_medidas = {
            "epochs": [],
            "training_times": [],
            "precisiones_bob": [],
            "precisiones_eve": [],
            "distancias_hamming_bob": [],
            "distancias_hamming_eve": [],
            "reconstrucciones_perfectas_bob": [],
            "reconstrucciones_perfectas_eve": [],
            "precisiones_bob_errado": [],
            "distancias_bob_errado": [],
            "reconstrucciones_perfectas_errado": []
        }
        
        time_0 = t.time()
        while epochs <= total_epochs:
            training_time = entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer, beta, gamma)
            
            res_list = evaluar(n_mensajes, bits, muestras, epochs, res_file_name)

            precisiones_bob = res_list[0]
            precisiones_eve = res_list[1]
            distancias_hamming_bob = res_list[2]
            distancias_hamming_eve = res_list[3]
            reconstrucciones_perfectas_bob = res_list[4]
            reconstrucciones_perfectas_eve = res_list[5]
            precisiones_bob_errado = res_list[6]
            distancias_bob_errado = res_list[7]
            reconstrucciones_perfectas_errado = res_list[8]

            diccionario_medidas["epochs"].append(epochs)
            diccionario_medidas["training_times"].append(training_time)
            
            diccionario_medidas["precisiones_bob"].append(precisiones_bob)
            diccionario_medidas["distancias_hamming_bob"].append(distancias_hamming_bob)
            diccionario_medidas["reconstrucciones_perfectas_bob"].append(reconstrucciones_perfectas_bob)
            
            diccionario_medidas["precisiones_eve"].append(precisiones_eve)
            diccionario_medidas["distancias_hamming_eve"].append(distancias_hamming_eve)
            diccionario_medidas["reconstrucciones_perfectas_eve"].append(reconstrucciones_perfectas_eve)

            diccionario_medidas["precisiones_bob_errado"].append(precisiones_bob_errado)
            diccionario_medidas["distancias_bob_errado"].append(distancias_bob_errado)
            diccionario_medidas["reconstrucciones_perfectas_errado"].append(reconstrucciones_perfectas_errado)

            epochs += step
        
        total_time = t.time() - time_0
        with open(res_file_name, "a") as f:
            f.write(f"\nTiempo total de ejecución (s) = {total_time}\n")
            f.write(f"Tiempo total de ejecución (mins) = {total_time/60}")

        draw_graph(diccionario_medidas, n_mensajes, nombre_figura)
    