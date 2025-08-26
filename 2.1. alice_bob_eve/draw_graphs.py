import matplotlib.pyplot as plt

def draw_graph(diccionario_medidas):

    list_epochs = diccionario_medidas["epochs"]
    list_training_times = diccionario_medidas["training_times"]

    list_precisiones_bob = diccionario_medidas["precisiones_bob"]
    list_precisiones_eve = diccionario_medidas["precisiones_eve"]
    list_precisiones_bob_errado = diccionario_medidas["precisiones_bob_errado"]
    
    list_distancias_hamming_bob = diccionario_medidas["distancias_hamming_bob"]
    list_distancias_hamming_eve = diccionario_medidas["distancias_hamming_eve"]
    list_distancias_hamming_bob_errado = diccionario_medidas["distancias_bob_errado"]

    list_reconstrucciones_perfectas_bob = diccionario_medidas["reconstrucciones_perfectas_bob"]
    list_reconstrucciones_perfectas_eve = diccionario_medidas["reconstrucciones_perfectas_eve"]
    list_reconstrucciones_perfectas_errado = diccionario_medidas["reconstrucciones_perfectas_errado"]
    
    plt.subplot(2, 2, 1)
    plt.plot(list_epochs, list_training_times, c="red", marker='o', markersize=3, markerfacecolor="red")
    plt.xlabel("Número de epochs")
    plt.ylabel("Duración del entrenamiento (s)")
    plt.title("Epochs - Entrenamiento")
    
    plt.subplot(2, 2, 2)
    plt.plot(list_epochs, list_precisiones_bob, c="red", label = "Prec Bob", marker='o', markersize=3, markerfacecolor="red")
    plt.plot(list_epochs, list_precisiones_bob_errado, c="blue", label = "Prec errada", linestyle="dashed", marker='o', markersize=3, markerfacecolor="blue")
    plt.plot(list_epochs, list_precisiones_eve, c="black", label = "Prec Eve", marker='o', linestyle="dotted", markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Media precisión descifrado")
    plt.title("Epochs - Precisión media")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(list_epochs, list_distancias_hamming_bob, c="red", label = "Dists. Bob", marker='o', markersize=3, markerfacecolor="red")
    plt.plot(list_epochs, list_distancias_hamming_bob_errado, c="blue", label = "Dists. erradas", linestyle="dashed", marker='o', markersize=3, markerfacecolor="blue")
    plt.plot(list_epochs, list_distancias_hamming_eve, c="black", label = "Dists. Eve", marker='o', linestyle="dotted", markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Media distancias de Hamming")
    plt.title("Epochs - Distancias de Hamming")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(list_epochs, list_reconstrucciones_perfectas_bob, c="red", label = "Perfectas Bob", marker='o', markersize=3, markerfacecolor="red")
    plt.plot(list_epochs, list_reconstrucciones_perfectas_errado, c="blue", label = "Pefectas errado", linestyle="dashed", marker='o', markersize=3, markerfacecolor="blue")
    plt.plot(list_epochs, list_reconstrucciones_perfectas_eve, c="black", label = "Perfectas Eve", linestyle="dotted", marker='o', markersize=3, markerfacecolor="black")
    plt.xlabel("Número de epochs")
    plt.ylabel("Reconstrucciones perfectas")
    plt.title("Epochs - Reconstrucciones perfectas")
    plt.legend()

    plt.show()



    