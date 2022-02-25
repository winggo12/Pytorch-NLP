import matplotlib.pyplot as plt
import pandas as pd
import time

def classification_report_csv(report, save_path):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(save_path, index = False)

def plot_final_acc_loss(train_test_dict):
    # plt_acc, plt_loss = plt.figure(1), plt.figure(2)
    for stage, acc_loss_dict in train_test_dict.items():
        for key, list in acc_loss_dict.items():
            epoches = [i for i in range(len(list))]
            plt_label = stage + " " + key
            if key == "acc" : plt.plot(epoches, list, label=plt_label)
            if key == "loss": plt.plot(epoches, list, label=plt_label)
        plt.legend()

    title = f"Accuracy_and_Loss"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc/Loss")
    plt.savefig(title + ".jpg")
    plt.show()
    plt.close()

def progress_bar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    progress = 'Progress: [%s%s] %d %%' % (arrow, spaces, percent)
    print('\r', 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='')

if __name__ == '__main__':
    a = {'acc': [1, 2, 3], 'loss': [0.1, 0.2, 0.3]}
    b = {'acc': [2, 4, 6], 'loss': [0.05, 0.1, 0.2]}
    c = {'train': a, 'test': b}

    plot_final_acc_loss(c)

    # for i in range(10):
    #     time.sleep(0.2)
    #     progress_bar(i, 10-1)
