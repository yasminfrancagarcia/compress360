from collections import defaultdict
import re 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

log = "checkpoints360/check_w-mse_cropWIdthOnly/5epocas/train360_wmse_crop_5epoc.log"

dados = defaultdict(lambda: {
    "loss": [],
    "mse": [],
    "bpp": [],
    "aux": []
})

epochs = []
losses = []
mse_losses = []
bpp_losses = []
aux_losses = []

with open(log, "r") as f:
    for line in f: 
        match = re.search(
            r"Train epoch (\d+):.*?"
            r"Loss:\s*([\d.]+)\s*\|"
            r"\s*MSE loss:\s*([\d.]+)\s*\|"
            r"\s*Bpp loss:\s*([\d.]+)\s*\|"
            r"\s*Aux loss:\s*([\d.]+)",
            line
        )

        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            mse = float(match.group(3))
            bpp = float(match.group(4))
            aux = float(match.group(5))

            dados[epoch]["loss"].append(loss)
            dados[epoch]["mse"].append(mse)
            dados[epoch]["bpp"].append(bpp)
            dados[epoch]["aux"].append(aux)

#calcula média por época
epochs = sorted(dados.keys())

loss = [
    sum(dados[e]["loss"]) / len(dados[e]["loss"])
    for e in epochs
]

mse = [
    sum(dados[e]["mse"]) / len(dados[e]["mse"])
    for e in epochs
]

bpp = [
    sum(dados[e]["bpp"]) / len(dados[e]["bpp"])
    for e in epochs
]

aux = [
    sum(dados[e]["aux"]) / len(dados[e]["aux"])
    for e in epochs
]


print("Épocas encontradas:", epochs)
print("Número de épocas:", len(epochs))


plt.figure(figsize=(10, 6))

plt.plot(epochs, loss, marker="o", label="loss")
plt.plot(epochs, mse, marker="o", label="MSE loss")
plt.plot(epochs, bpp, marker="o", label="Bpp loss")

plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.xlabel("épocas")
plt.ylabel("loss")
plt.title("Losses ao longo das épocas")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("convergencia_067_40epocas.png")
plt.show()