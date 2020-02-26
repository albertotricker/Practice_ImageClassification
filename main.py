import numpy as np
import multiprocessing
import tqdm
import os
import platform
import time
from lib.utils import read_data, plot_confusion_matrix, cross_validation
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn import svm
from tabulate import tabulate

print("  ______      _                      _   __        ")
print(" |  ____|    | |                    (_) /_/        ")
print(" | |__  __  _| |_ _ __ __ _  ___ ___ _  ___  _ __  ")
print(" |  __| \ \/ / __| '__/ _` |/ __/ __| |/ _ \| '_ \ ")
print(" | |____ >  <| |_| | | (_| | (_| (__| | (_) | | | |")
print(" |______/_/\_\\__|_|  \__,_|\___\___|_|\___/|_| |_|")
print("     | |      |  __ \                              ")
print("   __| | ___  | |__) |__ _ ___  __ _  ___  ___     ")
print("  / _` |/ _ \ |  _  // _` / __|/ _` |/ _ \/ __|    ")
print(" | (_| |  __/ | | \ \ (_| \__ \ (_| | (_) \__ \    ")
print("  \__,_|\___| |_|  \_\__,_|___/\__, |\___/|___/    ")
print("                                __/ |              ")
print("                               |___/  ")
print("")
print("Autor: Alberto Martín Martín\n")

exit_program = False
task = 0
while not exit_program:

    if task == 0:
        print("The next tasks have benn implemented:")
        print("1 - Classification evaluation: goodness measures and model parameters.")
        print("2 - Basic LBP implementation.")
        print("3 - Uniform LBP implementation.")
        print("4 - Combination of HOG and LBP descriptor.")
        print("5 - Pedestrian localization at different scales.")
        print("6 - Exit\n")
        task = input("Which task do you want to exec?(Enter the number of the task)")
        try:
            task = int(task)
            if task > 6:
                task = 0
            if platform.system() == "Linux":
                _ = os.system("clear")
            elif platform.system() == "Windows":
                _ = os.system("cls")
        except ValueError:
            task = 0
            if platform.system() == "Linux":
                _ = os.system("clear")
            elif platform.system() == "Windows":
                _ = os.system("cls")

    if task == 1:
        print("===========================================================================")
        print("Task 1: Classification evaluation: goodness measures and model parameters")
        print("===========================================================================\n")

        np.random.seed(23)
        print("Reading train data, HOG descriptor...")
        print("---------------------------------------------------------------------------")
        train, train_labels = read_data("data/train/", descriptor_type="hog")
        print("Reading test data, HOG descriptor...")
        print("---------------------------------------------------------------------------")
        test, test_labels = read_data("data/test/", descriptor_type="hog")
        shuffle = np.arange(train.shape[0] + test.shape[0])
        np.random.shuffle(shuffle)
        full_dataset = np.concatenate((train, test))[shuffle, :]
        full_labels = np.concatenate((train_labels, test_labels))[shuffle]
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")


        print("Part 1.1: Own implementation of a 10-fold cross-validation with parallelization using",
              "multiprocessing package and taking all the available cores. I have manually adjusted",
              "the params of the SVM classier with C=2, gamma=0.001 and kernel='rbf'.",
              "This combination of hyperparameters give a good result in term fo accuracy using",
              "HOG descriptor")
        execute = input("It takes like 4 or 5 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            k = 10
            kf = KFold(n_splits=k)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            cv = cross_validation(svm_classifier, full_dataset, full_labels, kf)
            cv_acc = list(tqdm.tqdm(pool.imap(cv.compute, range(k)), total=k))
            pool.close()
            print("")
            print("Mean accuracy:", np.mean(cv_acc))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")


        print("\nPart 1.2: Grid search of the SVM best parameters")
        execute = input("It takes like 20 or 25 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            parameters = {'kernel': ('linear', 'rbf'),
                          'C': [1, 5, 10],
                          'gamma': [0.1, 0.01, 0.001]}
            svm_classifier = svm.SVC()
            metric = 'accuracy'
            kf = KFold(n_splits=10)
            clf = GridSearchCV(estimator=svm_classifier,
                               param_grid=parameters,
                               scoring=metric,
                               n_jobs=multiprocessing.cpu_count(),
                               cv=kf,
                               verbose=5)
            clf.fit(full_dataset, full_labels)
            result_table = {}
            for params in clf.cv_results_['params']:
                for param, value in params.items():
                    if not param in result_table.keys():
                        result_table[param] = [value]
                    else:
                        result_table[param].append(value)
            result_table["Mean "+metric] = clf.cv_results_["mean_test_score"]
            result_table["Std "+metric] = clf.cv_results_["std_test_score"]
            result_table["Ranking"] = clf.cv_results_["rank_test_score"]
            print("")
            print("Results:")
            print(tabulate(result_table, headers="keys", tablefmt='orgtbl'))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")


        print("\nPart 1.3: Evaluate the goodness of the classifier under different type of measures",
              "like the precision, recall, F1 score or Kappa parameter. The classifier was trained",
              "with the same hyperparameters that in the part 1.1")
        execute = input("It takes like 2 or 3 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print('')
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            print("Training classier...\n")
            svm_classifier.fit(train, train_labels)
            prediction = svm_classifier.predict(test)
            acc = accuracy_score(test_labels, prediction)
            precision = precision_score(test_labels, prediction)
            recall = recall_score(test_labels, prediction)
            f1 = f1_score(test_labels, prediction)
            cm = confusion_matrix(test_labels, prediction)
            metrics = {"Accuracy": (acc,),
                       "Precision": (precision,),
                       "Recall": (recall,),
                       "F1 score": (f1,)}
            print("Metrics:")
            print(tabulate(metrics, headers="keys", tablefmt='orgtbl'))
            print('')
            conf_table = {"": ("Background", "Pedestrian"), "Background": cm[:, 0], "Pedestrian": cm[:, 1]}
            print("Confusion matrix:")
            print(tabulate(conf_table, headers="keys", tablefmt='orgtbl'))
            plot_confusion_matrix(test_labels, prediction, classes=np.array(["Background", "Pedestrian"]))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")
        task = 0

    elif task == 2:
        print("===========================================================================")
        print("Task 2: Basic LBP implementation")
        print("===========================================================================\n")

        np.random.seed(23)
        print("Reading train data, basic LBP descriptor...")
        print("---------------------------------------------------------------------------")
        train, train_labels = read_data("data/train/", descriptor_type="lbp", lbp_method="basic")
        print("Reading test data, basic LBP descriptor...")
        print("---------------------------------------------------------------------------")
        test, test_labels = read_data("data/test/", descriptor_type="lbp", lbp_method="basic")
        shuffle = np.arange(train.shape[0] + test.shape[0])
        np.random.shuffle(shuffle)
        full_dataset = np.concatenate((train, test))[shuffle, :]
        full_labels = np.concatenate((train_labels, test_labels))[shuffle]
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 2.1: Own implementation of a 10-fold cross-validation with parallelization using",
              "multiprocessing package and taking all the available cores. I have manually adjusted",
              "the params of the SVM classier with C=2, gamma=0.001 and kernel='rbf'.",
              "This combination of hyperparameters give a good result in term fo accuracy using",
              "the basic LBP descriptor")
        execute = input("It takes like 4 or 5 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            k = 10
            kf = KFold(n_splits=k)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            cv = cross_validation(svm_classifier, full_dataset, full_labels, kf)
            cv_acc = list(tqdm.tqdm(pool.imap(cv.compute, range(k)), total=k))
            pool.close()
            print("")
            print("Mean accuracy:", np.mean(cv_acc))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 2.2: Grid search of the SVM best parameters")
        execute = input("It takes like 10 or 15 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            parameters = {'kernel': ('linear', 'rbf'),
                          'C': [1, 5, 10],
                          'gamma': [0.1, 0.01, 0.001]}
            svm_classifier = svm.SVC()
            metric = 'accuracy'
            kf = KFold(n_splits=10)
            clf = GridSearchCV(estimator=svm_classifier,
                               param_grid=parameters,
                               scoring=metric,
                               n_jobs=multiprocessing.cpu_count(),
                               cv=kf,
                               verbose=5)
            clf.fit(full_dataset, full_labels)
            result_table = {}
            for params in clf.cv_results_['params']:
                for param, value in params.items():
                    if not param in result_table.keys():
                        result_table[param] = [value]
                    else:
                        result_table[param].append(value)
            result_table["Mean " + metric] = clf.cv_results_["mean_test_score"]
            result_table["Std " + metric] = clf.cv_results_["std_test_score"]
            result_table["Ranking"] = clf.cv_results_["rank_test_score"]
            print("")
            print("Results:")
            print(tabulate(result_table, headers="keys", tablefmt='orgtbl'))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 2.3: Evaluate the goodness of the classifier under different type of measures",
              "like the precision, recall, F1 score or Kappa parameter. The classifier was trained",
              "with the same hyperparameters that in the part 2.1")
        execute = input("It takes like 2 or 3 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print('')
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            print("Training classier...\n")
            svm_classifier.fit(train, train_labels)
            prediction = svm_classifier.predict(test)
            acc = accuracy_score(test_labels, prediction)
            precision = precision_score(test_labels, prediction)
            recall = recall_score(test_labels, prediction)
            f1 = f1_score(test_labels, prediction)
            cm = confusion_matrix(test_labels, prediction)
            metrics = {"Accuracy": (acc,),
                       "Precision": (precision,),
                       "Recall": (recall,),
                       "F1 score": (f1,)}
            print("Metrics:")
            print(tabulate(metrics, headers="keys", tablefmt='orgtbl'))
            print('')
            conf_table = {"": ("Background", "Pedestrian"), "Background": cm[:, 0], "Pedestrian": cm[:, 1]}
            print("Confusion matrix:")
            print(tabulate(conf_table, headers="keys", tablefmt='orgtbl'))
            plot_confusion_matrix(test_labels, prediction, classes=np.array(["Background", "Pedestrian"]))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")
        task = 0

    if task == 3:
        print("===========================================================================")
        print("Task 3: Uniform LBP implementation")
        print("===========================================================================\n")

        np.random.seed(23)
        print("Reading train data, uniform LBP descriptor...")
        print("---------------------------------------------------------------------------")
        train, train_labels = read_data("data/train/", descriptor_type="lbp", lbp_method="basic")
        print("Reading test data, uniform LBP descriptor...")
        print("---------------------------------------------------------------------------")
        test, test_labels = read_data("data/test/", descriptor_type="lbp", lbp_method="basic")
        shuffle = np.arange(train.shape[0] + test.shape[0])
        np.random.shuffle(shuffle)
        full_dataset = np.concatenate((train, test))[shuffle, :]
        full_labels = np.concatenate((train_labels, test_labels))[shuffle]
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 3.1: Own implementation of a 10-fold cross-validation with parallelization using",
              "multiprocessing package and taking all the available cores. I have manually adjusted",
              "the params of the SVM classier with C=2, gamma=0.001 and kernel='rbf'.",
              "This combination of hyperparameters give a good result in term fo accuracy using",
              "the uniform LBP descriptor")
        execute = input("It takes like 4 or 5 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            k = 10
            kf = KFold(n_splits=k)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            cv = cross_validation(svm_classifier, full_dataset, full_labels, kf)
            cv_acc = list(tqdm.tqdm(pool.imap(cv.compute, range(k)), total=k))
            pool.close()
            print("")
            print("Mean accuracy:", np.mean(cv_acc))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 3.2: Grid search of the SVM best parameters")
        execute = input("It takes like 10 or 15 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print("")
        if execute:
            parameters = {'kernel': ('linear', 'rbf'),
                          'C': [1, 5, 10],
                          'gamma': [0.1, 0.01, 0.001]}
            svm_classifier = svm.SVC()
            metric = 'accuracy'
            kf = KFold(n_splits=10)
            clf = GridSearchCV(estimator=svm_classifier,
                               param_grid=parameters,
                               scoring=metric,
                               n_jobs=multiprocessing.cpu_count(),
                               cv=kf,
                               verbose=5)
            clf.fit(full_dataset, full_labels)
            result_table = {}
            for params in clf.cv_results_['params']:
                for param, value in params.items():
                    if not param in result_table.keys():
                        result_table[param] = [value]
                    else:
                        result_table[param].append(value)
            result_table["Mean " + metric] = clf.cv_results_["mean_test_score"]
            result_table["Std " + metric] = clf.cv_results_["std_test_score"]
            result_table["Ranking"] = clf.cv_results_["rank_test_score"]
            print("")
            print("Results:")
            print(tabulate(result_table, headers="keys", tablefmt='orgtbl'))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")

        print("\nPart 3.3: Evaluate the goodness of the classifier under different type of measures",
              "like the precision, recall, F1 score or Kappa parameter. The classifier was trained",
              "with the same hyperparameters that in the part 3.1")
        execute = input("It takes like 2 or 3 minutes to compute, do you want to exec this?(Y/n)")
        execute = False if execute.lower() in ("n", "no") else True
        print('')
        if execute:
            C = 2
            gamma = 0.001
            kernel = "rbf"
            svm_classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
            print("Training classier...\n")
            svm_classifier.fit(train, train_labels)
            prediction = svm_classifier.predict(test)
            acc = accuracy_score(test_labels, prediction)
            precision = precision_score(test_labels, prediction)
            recall = recall_score(test_labels, prediction)
            f1 = f1_score(test_labels, prediction)
            cm = confusion_matrix(test_labels, prediction)
            metrics = {"Accuracy": (acc,),
                       "Precision": (precision,),
                       "Recall": (recall,),
                       "F1 score": (f1,)}
            print("Metrics:")
            print(tabulate(metrics, headers="keys", tablefmt='orgtbl'))
            print('')
            conf_table = {"": ("Background", "Pedestrian"), "Background": cm[:, 0], "Pedestrian": cm[:, 1]}
            print("Confusion matrix:")
            print(tabulate(conf_table, headers="keys", tablefmt='orgtbl'))
            plot_confusion_matrix(test_labels, prediction, classes=np.array(["Background", "Pedestrian"]))
            input("\nPress any key to continue...")
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")
        task = 0

    if task == 6:
        exit_program = True
        print("              _ _             _                                       _ ")
        print("     /\      | (_)           | |                                     | |")
        print("    /  \   __| |_  ___  ___  | |__  _   _ _ __ ___   __ _ _ __   ___ | |")
        print("   / /\ \ / _` | |/ _ \/ __| | '_ \| | | | '_ ` _ \ / _` | '_ \ / _ \| |")
        print("  / ____ \ (_| | | (_) \__ \ | | | | |_| | | | | | | (_| | | | | (_) |_|")
        print(" /_/    \_\__,_|_|\___/|___/ |_| |_|\__,_|_| |_| |_|\__,_|_| |_|\___/(_)")

        print('\n“Sé que tenéis miedo. Nos teméis a nosotros. Teméis el cambio. Yo no conozco el futuro.',
              "No he venido para deciros cómo acabará todo esto. Al contrario, he venido a deciros cómo va a comenzar.",
              "Voy a colgar el teléfono y luego voy a enseñarles a todos lo que vosotros no queréis que vean.",
              "Les enseñaré un mundo sin vosotros. Un mundo sin reglas y sin controles, sin limites ni fronteras.",
              "Un mundo donde cualquier cosa sea posible. Lo que hagamos después, es una decisión que dejo",
              "en vuestras manos.”")
        time.sleep(15)
        if platform.system() == "Linux":
            _ = os.system("clear")
        elif platform.system() == "Windows":
            _ = os.system("cls")
