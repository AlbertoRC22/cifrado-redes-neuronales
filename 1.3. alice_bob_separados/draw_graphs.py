import matplotlib.pyplot as plt

def draw_graph(diccionario_medidas, n_mensajes, nombre_figura):

    list_epochs = diccionario_medidas["epochs"]
    list_training_times = diccionario_medidas["training_times"]
    list_media_precision = diccionario_medidas["media_precision"]
    list_media_distancias = diccionario_medidas["media_distancias"]
    list_reconstrucciones_perfectas = diccionario_medidas["reconstrucciones_perfectas"]

    plt.subplot(2, 2, 1)
    plt.plot(list_epochs, list_training_times, c="red", marker='o', markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Duración entrenamiento (s)")
    plt.title("Entrenamiento - Epochs")
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    
    plt.subplot(2, 2, 2)
    plt.plot(list_epochs, list_media_precision, c="red", marker='o', markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Media precisión descifrado")
    plt.title("Precisión media - Epochs")
    plt.xlim(left = 0)

    plt.subplot(2, 2, 3)
    plt.plot(list_epochs, list_media_distancias, c="red", marker='o', markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Media distancias de Hamming")
    plt.title("Distancias de Hamming - Epochs")
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)

    plt.subplot(2, 2, 4)
    plt.plot(list_epochs, list_reconstrucciones_perfectas, c="red", marker='o', markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Reconstrucciones perfectas")
    plt.title("Reconstrucciones perfectas - Epochs")
    plt.xlim(left = 0)
    plt.ylim(0, n_mensajes)

    plt.subplots_adjust(left = 0.125, bottom = 0.11, right = 1.1, top = 0.9, wspace = 0.44, hspace = 0.45)
    plt.savefig(nombre_figura + ".png", bbox_inches='tight', dpi = 300)
    plt.close()



    