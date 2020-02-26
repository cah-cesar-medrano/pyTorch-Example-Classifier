from data_loader import loader as dl
from trainer import trainer, phaser
from tester import tester
import sys


def main():
    stdout = sys.stdout
    phaser_test = True

    sys.stdout = open('./out.log', 'w')

    classes, testset, testloader, trainloader, transform = dl()
    if(phaser_test):
        PATH = phaser(classes, testset, testloader, trainloader, transform)
    else:
        PATH = trainer(classes, testset, testloader, trainloader, transform)

    tester(testloader, classes, PATH)

    sys.stdout.close()
    sys.stdout = stdout


if __name__ == "__main__":
    main()
