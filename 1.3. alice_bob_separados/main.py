from train import entrenar
from eval import evaluar
from draw_graphs import draw_graph

if __name__ == '__main__':
    
    nombre_figura = "Figura 13 - Entrenamiento separado"
    res_file_name = "Resultados_13.txt"
    with open(res_file_name, "w") as f:
        f.write("-- RESULTADOS DEL EXPERIMENTO 1.3. --\n\n")

    n_mensajes = 10000
    bits = 32
    batch_size = 64
    adam_optimizer = 0.001
    muestras = 10

    epochs = 500
    step = 500
    total_epochs = 4000

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
        f.write(f"Adam optimizer learning rate = {adam_optimizer}\n")
        f.write(f"Epochs totales = {total_epochs}\n")
        f.write(f"Epochs iniciales = {epochs}\n")
        f.write("\n")
    
    while epochs <= total_epochs:
        training_time = entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer)
        res_list = evaluar(bits, muestras, res_file_name, epochs)
        
        media_precision = res_list[0]
        media_distancias = res_list[1]
        reconstrucciones_perfectas = res_list[2]

        diccionario_medidas["epochs"].append(epochs)
        diccionario_medidas["training_times"].append(training_time)
        diccionario_medidas["media_precision"].append(media_precision)
        diccionario_medidas["media_distancias"].append(media_distancias)
        diccionario_medidas["reconstrucciones_perfectas"].append(reconstrucciones_perfectas)

        epochs += step

    draw_graph(diccionario_medidas, n_mensajes, nombre_figura)
    