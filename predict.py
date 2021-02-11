
def main_predict(desired_outputs, test, network):
    correct = 0
    for i in range(len(test)):
        network.setInput(test[i])
        network.feedForward()
        network.backPropagate(desired_outputs[i])
        if (res[0] > 0.5 and desired_outputs[i][0] == 1) or \
                (res[0] < 0.5 and desired_outputs[i][0] == 0):
                    correct += 1
        print("Network predicted \033[1m{}\033[0m for \033[1m{}\033[0m input".format(res[0], desired_outputs[i][0]))
    print("Correctly predicted {} out of {} test samples".format(correct, len(test)))
    print("Network Accuracy : {}%".format((correct / len(test)) * 100))
