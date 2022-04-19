import powerlaw as po
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

def read_file(fname):
    data = []
    file = open(fname, 'r')
    # reading each line
    for line in file:
        # reading each word
        for word in line.split():
            fix_word = word.strip(",.\":")
            data.append(fix_word)
    data.remove(data[0])
    return data

def dicts(data):
    wc = Counter(data)
    vocab = [w for w,c in wc.most_common(10000)]
    word2count = {w:c for w,c in wc.most_common(10000)}
    word2ID = {word:i+1 for i,word in enumerate(vocab)}
    word2prob = {word:float(c)/len(data) for word,c in wc.most_common(10000)}

    return word2count, word2ID, word2prob

def find_params(data,flag):
    if flag:
        ks = 1000
        best_alpha = 0
        best_xmin = 0
        for xmin in range(1,100):
            alpha = 1 + len(data)/np.sum(np.log(data/xmin))
            new_ks = po.power_law_ks_distance(data,alpha,xmin)
            if new_ks < ks:
                ks = new_ks
                best_xmin = xmin
                best_alpha = alpha
        return best_xmin, best_alpha, ks

def pdf(xmin,alpha,x):
    C = alpha-1
    if x<xmin:
        return 0
    return C*np.power(x,-alpha)

def cdf(xmin,alpha,x):
    return -np.power(x,-alpha+1)+np.power(xmin,-alpha+1)

def ccdf(xmin,alpha,y):
    return np.power(np.power(xmin,-alpha+1)-y, 1 / (1-alpha))

def calculate_goodnes(xmin, alpha, ks, N):
    print(f"Calculate goodness-of-fit, with K = 100:\n")
    np.random.seed(2)
    K = 100
    counter = 0
    for _ in range(K):
        sample_N = np.array([np.random.rand(N)])
        data = np.array([np.floor(ccdf(xmin,alpha,i)) for i in sample_N])
        a,b,test_ks  = find_params(data,1)
        print(f"step {_+1}:\tbest_xmin: {a}\tks: {test_ks}")
        if ks < test_ks:
            counter += 1
    print(f"\nThe goodness-of-fit is: {counter/K}")


if __name__ == '__main__':
    # Read data
    file = read_file("The_Hunger_Games.txt")
    word2count, word2ID, word2prob = dicts(file)
    data = np.array([word2ID[word] for word in file])

    # Calculate ecdf
    ecdf = ECDF(data)
    # Find best xmin and alpha that powerlaw fit the data
    xmin, alpha, ks = find_params(data, 1)
    print(f"The params of our powerlaw is:\txmin: {xmin},\talpha: {alpha}\n")

    # Calculate for plots
    X = np.array(list(word2ID.values()))
    Y_ecdf = np.array(ecdf(X))
    Y_cdf = np.array([cdf(xmin, alpha, x) for x in X])
    Y_pdf = np.array([pdf(xmin, alpha, x) for x in list(word2ID.values())])
    Y_probs = np.array([word2prob[w] for w in list(word2ID.keys())])

    # Plot histogram
    plt.xlim(0,1000)
    plt.hist(data, bins=400, density=True, edgecolor="black")
    plt.xlabel("x axis")
    plt.ylabel("Probability axis")
    plt.title("Histogram")
    plt.show()

    # Plot ECDF and CDF from powerlaw fitted
    plt.plot(X, Y_ecdf, color='red')
    plt.plot(X, Y_cdf)
    plt.xlabel("x axis")
    plt.ylabel("Probability axis")
    plt.title("ECDF and CDF")
    plt.legend(["ECDF", " real CDF"])
    plt.show()

    # Plot the probs of the random variable and pdf from powerlaw fitted
    plt.xlim(0, 80)
    plt.scatter(X,Y_probs, s=10)
    plt.plot(X, Y_pdf, color='red')
    plt.xlabel("x axis")
    plt.ylabel("Probability axis")
    plt.title("Probabilities of the random variable and PDF")
    plt.legend(["Probabilities of the random variable", " real PDF"])
    plt.show()

    calculate_goodnes(xmin, alpha, ks, len(data))
