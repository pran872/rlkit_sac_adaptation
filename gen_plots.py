import pandas as pd
import matplotlib.pyplot as plt

def plt_it(x, ys):
    fig, ax = plt.subplots(1, 3)
    for no, (y_label, y) in enumerate(ys.items()):
        ax[no].plot(x, y)
        ax[no].set(xlabel='Epoch', ylabel=y_label)
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("data/name-of-experiment/e500_2024_12_07_15_37_42_0000--s-0/progress.csv")
    epoch = df['Epoch']

    ys = {'Average Returns': df['eval/Average Returns'],
          'QF1 Loss': df['trainer/QF1 Loss'],
          'QF2 Loss': df['trainer/QF2 Loss']}
    plt_it(epoch, ys)

