from util import *
from sklearn import svm

if __name__ == '__main__':
    voca = load_voca('./ex6/vocab.txt')
    sample_email_file = open('./ex6/emailSample1.txt', mode='r')
    sample_email_content = ''
    for line in sample_email_file.readlines():
        line = line.strip('\n')
        sample_email_content += line
    word_index = preprocess_email(sample_email_content, voca)
    print(word_index)
    features = email_features(word_index)
    print('Length of feature vector:', len(features))
    print('Non-zero entries:', np.sum(features > 0))

    model = svm.LinearSVC(C=0.1)
    data = read_matlab('./ex6/spamTrain.mat')
    X = data['X']
    y = data['y'].ravel()
    data_test = read_matlab('./ex6/spamTest.mat')
    X_test = data_test['Xtest']
    y_test = data_test['ytest'].ravel()
    print('\033[1;33mBegin to train SVM, please wait for few seconds.\033[0m')
    model.fit(X, y)
    predict_train = model.predict(X)
    predict = model.predict(X_test)
    print('\033[1;32mThe accuracy on train dataset is: {}%.\033[0m'.format(np.mean(y == predict_train) * 100))
    print('\033[1;32mThe accuracy on test dataset is: {}%.\033[0m'.format(np.mean(y_test == predict) * 100))
    coef = model.coef_.ravel()
    idx = coef.argsort()[::-1]
    vocab_list = list(voca)

    print('Top predictors of spam:')
    for i in range(15):
        print("{0:<15s} ({1:f})".format(vocab_list[idx[i]], coef[idx[i]]))
