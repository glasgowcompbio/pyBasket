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

from FuncVelov3 import validate_workflow, train, plots, heldout_testF, \
    set_best_params_from_paper
from NetVelo import get_network
from TestLoadData import prep_data

# import sys
# sys.setrecursionlimit(1000000)
# import warnings
# warnings.filterwarnings("ignore")

# torch.set_num_threads(64)
# matplotlib.use('Agg')

def main():
    train_arg_parser = argparse.ArgumentParser()
    train_arg_parser.add_argument("--drug", type=str, default='Erlotinib',
                                  help='input drug to train a model')
    train_arg_parser.add_argument("--data_root", type=str, default='../Data/',
                                  help="path to molecular and pharmacological data")
    train_arg_parser.add_argument("--save_logs", type=str, default='./VelodromeTest/logs/',
                                  help='path of folder to write log')
    train_arg_parser.add_argument("--save_models", type=str, default='./VelodromeTest/models/',
                                  help='folder for saving model')
    train_arg_parser.add_argument("--save_results", type=str, default='./VelodromeTest/results/',
                                  help='folder for saving model')
    train_arg_parser.add_argument("--hd", type=int, default=2, help='strcuture of the network')
    train_arg_parser.add_argument("--bs", type=int, default=64, help='strcuture of the network')
    train_arg_parser.add_argument("--ldr", type=float, default=0.5, help='dropout')
    train_arg_parser.add_argument("--wd", type=float, default=0.5, help='weight decay')
    train_arg_parser.add_argument("--wd1", type=float, default=0.1, help='weight decay 1')
    train_arg_parser.add_argument("--wd2", type=float, default=0.1, help='weight decay 2')
    train_arg_parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    train_arg_parser.add_argument("--lr1", type=float, default=0.005, help='learning rate 1')
    train_arg_parser.add_argument("--lr2", type=float, default=0.005, help='learning rate 2')
    train_arg_parser.add_argument("--lam1", type=float, default=0.005, help='lambda 1')
    train_arg_parser.add_argument("--lam2", type=float, default=0.005, help='lambda 2')
    train_arg_parser.add_argument("--epoch", type=int, default=30, help='number of epochs')
    train_arg_parser.add_argument("--seed", type=int, default=42, help='set the random seed')
    train_arg_parser.add_argument('--gpu', type=int, default=0, help='set using GPU or not')
    train_arg_parser.add_argument('--best_params', type=int, default=0,
                                  help='use best parameters from paper')

    args = train_arg_parser.parse_args()
    set_best_params_from_paper(args, out_dir='VelodromeTest')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = [args.save_logs, args.save_models, args.save_results]

    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path + '/args.txt', 'w') as f:
            f.write(str(args))

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.drug == "Docetaxel":
        X_tr, Y_tr, X_U, TCGA_PRAD, TCGA_KIRC, gCSI_exprs_drug, gCSI_aac_drug = prep_data(args)
    else:
        X_tr, Y_tr, X_U, TCGA_PRAD, TCGA_KIRC, gCSI_exprs_drug, gCSI_aac_drug = prep_data(args)

    X_tr1 = X_tr[0]
    Y_tr1 = Y_tr[0]
    X_tr2 = X_tr[1]
    Y_tr2 = Y_tr[1]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X_tr1, Y_tr1, test_size=0.2,
                                                            random_state=args.seed, shuffle=True)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_tr2, Y_tr2, test_size=0.1,
                                                            random_state=args.seed, shuffle=True)
    # XU_train, XU_test = train_test_split(X_U, test_size=0.3, random_state=42, Shuffle=True)

    best_pr = 0

    loss_fun = torch.nn.MSELoss()
    total_val = []
    total_aac = []

    train_loss = []
    consistency_loss = []
    covariance_loss = []
    train_pr1 = []
    train_pr2 = []
    val_loss = []
    val_pr = []

    X_train = np.concatenate((X1_train, X2_train, X_U), axis=0)
    X_val = np.concatenate((X1_test, X2_test), axis=0)
    y_val = np.concatenate((y1_test, y2_test), axis=0)

    scaler = sk.StandardScaler()
    scaler.fit(X_train)
    X1_train_N = scaler.transform(X1_train)
    X2_train_N = scaler.transform(X2_train)
    X_U_N = scaler.transform(X_U)

    PRAD_N = torch.FloatTensor(scaler.transform(TCGA_PRAD.values))
    KIRC_N = torch.FloatTensor(scaler.transform(TCGA_KIRC.values))
    gCSI_N = torch.FloatTensor(scaler.transform(gCSI_exprs_drug.values))

    TX_val_N = torch.FloatTensor(scaler.transform(X_val))
    Ty_val = torch.FloatTensor(y_val)

    train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train_N),
                                                   torch.FloatTensor(y1_train))
    trainLoader_1 = torch.utils.data.DataLoader(dataset=train1Dataset, batch_size=args.bs,
                                                shuffle=True, num_workers=1)

    train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train_N),
                                                   torch.FloatTensor(y2_train))
    trainLoader_2 = torch.utils.data.DataLoader(dataset=train2Dataset, batch_size=args.bs,
                                                shuffle=True, num_workers=1)

    trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U_N))
    trainULoader = torch.utils.data.DataLoader(dataset=trainUDataset, batch_size=args.bs,
                                               shuffle=True, num_workers=1)

    model, predict_1, predict_2 = get_network(args, X1_train_N)

    opt = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    opt1 = optim.Adagrad(predict_1.parameters(), lr=args.lr1, weight_decay=args.wd1)
    opt2 = optim.Adagrad(predict_2.parameters(), lr=args.lr2, weight_decay=args.wd2)

    train_pred = []
    w1 = []
    w2 = []

    for ite in tqdm(range(args.epoch)):

        pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(
            args, model, predict_1, predict_2, loss_fun, opt, opt1, opt2,
            trainLoader_1, trainLoader_2, trainULoader)

        train_loss.append(pred_loss + coral_loss + con_loss)
        train_loss.append(pred_loss + con_loss)
        consistency_loss.append(con_loss)
        covariance_loss.append(coral_loss)
        train_pr1.append(epoch_pr1)
        train_pr2.append(epoch_pr2)

        w1.append(loss1)
        w2.append(loss2)

        ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
        epoch_val_loss, epoch_Val_pr, _ = validate_workflow(args, model, predict_1, predict_2,
                                                            ws, loss_fun, TX_val_N, Ty_val)
        val_loss.append(epoch_val_loss)
        val_pr.append(epoch_Val_pr)

        f = open(os.path.join(args.save_logs, 'args.txt'), mode='a')
        f.write('iteration:{}, train loss:{}\n'.format(ite, train_loss))
        f.write('iteration:{}, validation epoch loss:{}\n'.format(ite, epoch_val_loss))
        f.close()

        if epoch_Val_pr > best_pr:
            best_pr = epoch_Val_pr
            f = open(os.path.join(args.save_results, 'Best_val.txt'), mode='a')
            f.write('iteration:{}, best validation correlation:{}\n'.format(ite, best_pr))
            f.close()
            torch.save(model.state_dict(), os.path.join(args.save_models, 'Best_Model.pt'))
            torch.save(predict_1.state_dict(), os.path.join(args.save_models, 'Best_Pred1.pt'))
            torch.save(predict_2.state_dict(), os.path.join(args.save_models, 'Best_Pred2.pt'))

    plots(args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss,
          val_pr)
    model.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Model.pt')))
    predict_1.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred1.pt')))
    predict_2.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred2.pt')))

    model.eval()
    predict_1.eval()
    predict_2.eval()

    _, _, preds = validate_workflow(args, model, predict_1, predict_2,
                                    [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))],
                                    loss_fun, TX_val_N, Ty_val)
    total_val.append(preds.detach().numpy().flatten())
    total_aac.append(Ty_val.detach().numpy())

    model.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Model.pt')))
    predict_1.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred1.pt')))
    predict_2.load_state_dict(torch.load(os.path.join(args.save_models, 'Best_Pred2.pt')))

    final_pred = list(itertools.chain.from_iterable(total_val))
    final_labels = list(itertools.chain.from_iterable(total_aac))
    f = open(os.path.join(args.save_results, 'Total_val.txt'), mode='a')
    f.write('Total validation Pearson:{}\n'.format(pearsonr(final_pred, final_labels)))
    f.write('Total validation Spearman:{}\n'.format(spearmanr(final_pred, final_labels)))
    f.write('Total validation Kendall:{}\n'.format(kendalltau(final_pred, final_labels)))
    f.write('---------------------------------\n')
    f.close()

    ws = [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))]
    r_gCSI_N, pval1, sr_gCSI_N, pval2, pred_PRAD, pred_KIRC = heldout_testF(
        args, model, predict_1, predict_2, ws, PRAD_N, KIRC_N, gCSI_N, gCSI_aac_drug)

    f = open(os.path.join(args.save_results, 'gCSI-nonSolid.txt'), mode='a')
    f.write('Test Pearson gCSI:{}\n'.format(r_gCSI_N))
    f.write('P value Pearson gCSI:{}\n'.format(pval1))
    f.write('---------------------------------\n')
    f.close()

    np.savetxt(os.path.join(args.save_results, 'TCGA_PRAD.txt'), pred_PRAD)

    np.savetxt(os.path.join(args.save_results, 'TCGA_KIRC.txt'), pred_KIRC)


if __name__ == "__main__":
    main()
