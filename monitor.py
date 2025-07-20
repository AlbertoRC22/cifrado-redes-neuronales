import numpy as np
import matplotlib.pyplot as plt
import os

# Inicializar historial con la nueva métrica
def inicializar_historial():
    return {
        'bob': [],
        'charlie': [],
        'eve': [],
        'non_dst': [],
        'total': [],
        'mejor_perdida': np.inf
    }

# Registrar pérdidas por época
def registrar_pérdidas(historial, loss_bob, loss_charlie, loss_eve, loss_non_dst, loss_total):
    historial['bob'].append(loss_bob)
    historial['charlie'].append(loss_charlie)
    historial['eve'].append(loss_eve)
    historial['non_dst'].append(loss_non_dst)
    historial['total'].append(loss_total)

# Guardar modelos si mejora pérdida ajustada
def guardar_si_mejora(loss_total, historial, alice, bob, charlie):
    if loss_total < historial['mejor_perdida']:
        historial['mejor_perdida'] = loss_total
        alice.save('modelo_alice.keras')
        bob.save('modelo_bob.keras')
        charlie.save('modelo_charlie.keras')
        print("💾 Modelos guardados (mejor pérdida ajustada).")

# Generar gráfico final con pérdida no destinatario
def graficar_historial(historial, ruta='grafico_perdidas.png'):
    print("\n📈 Generando gráfico de pérdidas...")
    plt.figure(figsize=(10, 6))
    plt.plot(historial['bob'], label='Bob')
    plt.plot(historial['charlie'], label='Charlie')
    plt.plot(historial['eve'], label='Eve')
    plt.plot(historial['non_dst'], label='No Destinatario ❌', linestyle='dashdot')
    plt.plot(historial['total'], label='Pérdida Total 🎯', linestyle='--', linewidth=2)

    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Evolución de pérdidas por época")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ruta)
    print(f"✅ Gráfico guardado como '{ruta}'")
