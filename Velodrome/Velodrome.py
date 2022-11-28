import argparse
import itertools
import os
import random

import numpy as np
import sklearn.preprocessing as sk
import torch.optim as optim
import torch.utils.data
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from FuncVelov3 import validate_workflow, train, plots, \
    heldout_test, heldout_testv3, \
    set_best_params_from_paper, heldout_testF
from NetVelo import get_network
from LoadData import prep_train_data, prep_test_data


# import sys
# sys.setrecursionlimit(1000000)
# import warnings
# warnings.filterwarnings("ignore")

# torch.set_num_threads(64)
# matplotlib.use('Agg')


def main():
    args = parse_args()

    ### Fix random seed ###
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ### Create output dirs if not exist ###
    paths = [args.save_logs, args.save_models, args.save_results]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path + '/args.txt', 'w') as f:
            f.write(str(args))

    ### Data extraction ###
    if args.stage == 'train':
        if args.drug == "Docetaxel":
            X_tr, Y_tr, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, X_U = prep_train_data(args)
        else:
            X_tr, Y_tr, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, X_ts_3, Y_ts_3, X_U = prep_train_data(args)

    elif args.stage == 'test':
        if args.drug == "Docetaxel":
            X_tr, Y_tr, X_U, TCGA_PRAD, TCGA_KIRC, gCSI_exprs_drug, gCSI_aac_drug = prep_test_data(
                args)
        else:
            X_tr, Y_tr, X_U, TCGA_PRAD, TCGA_KIRC, gCSI_exprs_drug, gCSI_aac_drug = prep_test_data(
                args)

    # make training and test sets
    TX_val_N, Ty_val, X1_train_N, scaler, train_loader_1, train_loader_2, train_loader_U = \
        make_datasets(args, X_tr, Y_tr, X_U)

    ### Define models and optimisers ###
    model, predict_1, predict_2 = get_network(args, X1_train_N)
    opt = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    opt1 = optim.Adagrad(predict_1.parameters(), lr=args.lr1, weight_decay=args.wd1)
    opt2 = optim.Adagrad(predict_2.parameters(), lr=args.lr2, weight_decay=args.wd2)

    # Training loop starts here ###
    loss_fun = torch.nn.MSELoss()
    train_loss = []
    consistency_loss = []
    covariance_loss = []
    train_pr1 = []
    train_pr2 = []
    val_loss = []
    val_pr = []
    w1 = []
    w2 = []
    best_pr = 0
    for ite in tqdm(range(args.epoch)):

        pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(
            args, model, predict_1, predict_2, loss_fun, opt, opt1, opt2,
            train_loader_1, train_loader_2, train_loader_U)

        train_loss.append(pred_loss + coral_loss + con_loss)
        train_loss.append(pred_loss + con_loss)
        consistency_loss.append(con_loss)
        covariance_loss.append(coral_loss)
        train_pr1.append(epoch_pr1)
        train_pr2.append(epoch_pr2)

        w1.append(loss1)
        w2.append(loss2)
        ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
        epoch_val_loss, epoch_val_pr, _ = validate_workflow(args, model, predict_1, predict_2,
                                                            ws, loss_fun, TX_val_N, Ty_val)
        val_loss.append(epoch_val_loss)
        val_pr.append(epoch_val_pr)

        log_training(args, ite, epoch_val_loss, train_loss)
        if epoch_val_pr > best_pr:
            best_pr = epoch_val_pr
            log_best_model(args, best_pr, ite)
            save_best_model(args, model, predict_1, predict_2)

    plots(args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss,
          val_pr)
    load_model_state_dicts(args, model, predict_1, predict_2)
    model.eval()
    predict_1.eval()
    predict_2.eval()

    ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
    _, _, preds = validate_workflow(args, model, predict_1, predict_2, ws,
                                    loss_fun, TX_val_N, Ty_val)

    load_model_state_dicts(args, model, predict_1, predict_2)
    correlation_report(Ty_val, args, preds)

    if args.stage == 'train':

        if args.drug == "Docetaxel":
            ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
            test_1, test_s1, test_roc2, test_aupr2 = heldout_test(
                args, model, predict_1, predict_2, ws, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, scaler)
            final_report_train_Docetaxel(args, test_1, test_aupr2, test_roc2, test_s1)

        else:
            ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
            test_1, test_s1, test_roc2, test_aupr2, test_roc3, test_aupr3 = heldout_testv3(
                args, model, predict_1, predict_2, ws,
                X_ts_1, Y_ts_1, X_ts_2, Y_ts_2, X_ts_3, Y_ts_3, scaler)
            final_report_train_others(args, test_1, test_aupr2, test_aupr3, test_roc2, test_roc3,
                                      test_s1)

    elif args.stage == 'test':

        PRAD_N = torch.FloatTensor(scaler.transform(TCGA_PRAD.values))
        KIRC_N = torch.FloatTensor(scaler.transform(TCGA_KIRC.values))
        gCSI_N = torch.FloatTensor(scaler.transform(gCSI_exprs_drug.values))

        ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
        r_gCSI_N, pval1, sr_gCSI_N, pval2, pred_PRAD, pred_KIRC = heldout_testF(
            args, model, predict_1, predict_2, ws, PRAD_N, KIRC_N, gCSI_N, gCSI_aac_drug)
        final_report_test(args, pred_KIRC, pred_PRAD, pval1, r_gCSI_N)


def parse_args():
    parser = argparse.ArgumentParser()
    choices = ['Docetaxel', 'Gemcitabine', 'Erlotinib', 'Paclitaxel']
    parser.add_argument("--drug", type=str, choices=choices,
                        help='input drug to train a model', required=True)
    parser.add_argument("--data_root", type=str, default='../Data/',
                        help="path to molecular and pharmacological data")
    parser.add_argument("--save_logs", type=str, default='./logs/',
                        help='path of folder to write log')
    parser.add_argument("--save_models", type=str, default='./models/',
                        help='folder for saving model')
    parser.add_argument("--save_results", type=str, default='./results/',
                        help='folder for saving model')
    parser.add_argument("--hd", type=int, default=2, help='strcuture of the network')
    parser.add_argument("--bs", type=int, default=64, help='strcuture of the network')
    parser.add_argument("--ldr", type=float, default=0.5, help='dropout')
    parser.add_argument("--wd", type=float, default=0.5, help='weight decay')
    parser.add_argument("--wd1", type=float, default=0.1, help='weight decay 1')
    parser.add_argument("--wd2", type=float, default=0.1, help='weight decay 2')
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    parser.add_argument("--lr1", type=float, default=0.005, help='learning rate 1')
    parser.add_argument("--lr2", type=float, default=0.005, help='learning rate 2')
    parser.add_argument("--lam1", type=float, default=0.005, help='lambda 1')
    parser.add_argument("--lam2", type=float, default=0.005, help='lambda 2')
    parser.add_argument("--epoch", type=int, default=30, help='number of epochs')
    parser.add_argument("--seed", type=int, default=42, help='set the random seed')
    parser.add_argument('--best_params', action='store_true',
                        help='use best parameters from paper')
    choices = ['train', 'test']
    parser.add_argument("--stage", choices=choices, required=True)
    args = parser.parse_args()
    set_best_params_from_paper(args)
    return args


def make_datasets(args, X_tr, Y_tr, X_U):
    ### Split into training and test sets ###
    X_tr1 = X_tr[0]
    Y_tr1 = Y_tr[0]
    X_tr2 = X_tr[1]
    Y_tr2 = Y_tr[1]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_tr1, Y_tr1, test_size=0.2,
                                                            random_state=args.seed, shuffle=True)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_tr2, Y_tr2, test_size=0.1,
                                                            random_state=args.seed, shuffle=True)
    X_train = np.concatenate((X1_train, X2_train, X_U), axis=0)
    X_val = np.concatenate((X1_test, X2_test), axis=0)
    y_val = np.concatenate((y1_test, y2_test), axis=0)

    ### Scale input ###
    scaler = sk.StandardScaler()
    scaler.fit(X_train)
    X1_train_N = scaler.transform(X1_train)
    X2_train_N = scaler.transform(X2_train)
    X_U_N = scaler.transform(X_U)
    TX_val_N = torch.FloatTensor(scaler.transform(X_val))
    Ty_val = torch.FloatTensor(y_val)

    ### Create PyTorch datasets ###
    train_dataset_1 = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train_N),
                                                     torch.FloatTensor(y1_train))
    train_loader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1, batch_size=args.bs,
                                                 shuffle=True, num_workers=1)
    train_dataset_2 = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train_N),
                                                     torch.FloatTensor(y2_train))
    train_loader_2 = torch.utils.data.DataLoader(dataset=train_dataset_2, batch_size=args.bs,
                                                 shuffle=True, num_workers=1)
    train_dataset_U = torch.utils.data.TensorDataset(torch.FloatTensor(X_U_N))
    train_loader_U = torch.utils.data.DataLoader(dataset=train_dataset_U, batch_size=args.bs,
                                                 shuffle=True, num_workers=1)
    return TX_val_N, Ty_val, X1_train_N, scaler, train_loader_1, train_loader_2, train_loader_U


def log_best_model(args, best_pr, ite):
    f = open(os.path.join(args.save_results, 'Best_val.txt'), mode='a')
    f.write('iteration:{}, best validation correlation:{}\n'.format(ite, best_pr))
    f.close()


def save_best_model(args, model, predict_1, predict_2):
    torch.save(model.state_dict(), os.path.join(args.save_models, 'Best_Model.pt'))
    torch.save(predict_1.state_dict(), os.path.join(args.save_models, 'Best_Pred1.pt'))
    torch.save(predict_2.state_dict(), os.path.join(args.save_models, 'Best_Pred2.pt'))


def log_training(args, ite, epoch_val_loss, train_loss):
    f = open(os.path.join(args.save_logs, 'args.txt'), mode='a')
    f.write('iteration:{}, train loss:{}\n'.format(ite, train_loss))
    f.write('iteration:{}, validation epoch loss:{}\n'.format(ite, epoch_val_loss))
    f.close()


def load_model_state_dicts(args, model, predict_1, predict_2):
    model.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Model.pt')))
    predict_1.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred1.pt')))
    predict_2.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred2.pt')))


def correlation_report(Ty_val, args, preds):
    total_val = []
    total_aac = []
    total_val.append(preds.detach().numpy().flatten())
    total_aac.append(Ty_val.detach().numpy())
    final_pred = list(itertools.chain.from_iterable(total_val))
    final_labels = list(itertools.chain.from_iterable(total_aac))
    f = open(os.path.join(args.save_results, 'Total_val.txt'), mode='a')
    f.write('Total validation Pearson:{}\n'.format(pearsonr(final_pred, final_labels)))
    f.write('Total validation Spearman:{}\n'.format(spearmanr(final_pred, final_labels)))
    f.write('Total validation Kendall:{}\n'.format(kendalltau(final_pred, final_labels)))
    f.write('---------------------------------\n')
    f.close()


def final_report_train_Docetaxel(args, test_1, test_aupr2, test_roc2, test_s1):
    f = open(os.path.join(args.save_results, 'Target.txt'), mode='a')
    f.write('Test Pearson gCSI:{}\n'.format(test_1))
    f.write('Test Spearman gCSI:{}\n'.format(test_s1))
    f.write('---------------------------------\n')
    f.write('ROC Patient:{}\n'.format(test_roc2))
    f.write('AUPR Patient:{}\n'.format(test_aupr2))
    f.write('---------------------------------\n')
    f.close()


def final_report_train_others(args, test_1, test_aupr2, test_aupr3, test_roc2, test_roc3, test_s1):
    f = open(os.path.join(args.save_results, 'Target.txt'), mode='a')
    f.write('Test Pearson gCSI:{}\n'.format(test_1))
    f.write('Test Spearman gCSI:{}\n'.format(test_s1))
    f.write('---------------------------------\n')
    f.write('ROC Patient:{}\n'.format(test_roc2))
    f.write('AUPR Patient:{}\n'.format(test_aupr2))
    f.write('---------------------------------\n')
    f.write('ROC PDX:{}\n'.format(test_roc3))
    f.write('AUPR PDX:{}\n'.format(test_aupr3))
    f.close()


def final_report_test(args, pred_KIRC, pred_PRAD, pval1, r_gCSI_N):
    f = open(os.path.join(args.save_results, 'gCSI-nonSolid.txt'), mode='a')
    f.write('Test Pearson gCSI:{}\n'.format(r_gCSI_N))
    f.write('P value Pearson gCSI:{}\n'.format(pval1))
    f.write('---------------------------------\n')
    f.close()
    np.savetxt(os.path.join(args.save_results, 'TCGA_PRAD.txt'), pred_PRAD)
    np.savetxt(os.path.join(args.save_results, 'TCGA_KIRC.txt'), pred_KIRC)


if __name__ == "__main__":
    main()
