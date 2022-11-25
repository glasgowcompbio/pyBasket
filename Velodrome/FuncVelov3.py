import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# matplotlib.use('Agg')

def train(args, model, predict_1, predict_2, loss, opt, opt1, opt2, dataLoader_tr1, dataLoader_tr2,
          dataULoader_tr):
    pred_cost = []
    coral_cost = []
    con_cost = []
    epoch_PR1 = []
    epoch_PR2 = []
    loss1 = []
    loss2 = []

    model.train()
    predict_1.train()
    predict_2.train()

    for i, data in enumerate(zip(dataLoader_tr1, cycle(dataLoader_tr2))):
        data_s1 = data[0]
        data_s2 = data[1]
        xs1 = data_s1[0]
        ys1 = data_s1[1].view(-1, 1)
        xs2 = data_s2[0]
        ys2 = data_s2[1].view(-1, 1)

        fx_1 = model(xs1)
        fx_2 = model(xs2)

        pred_1 = predict_1(fx_1)
        pred_2 = predict_2(fx_2)

        opt.zero_grad()
        opt1.zero_grad()
        opt2.zero_grad()

        # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
        # https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085
        ls1 = loss(pred_1, ys1)
        ls2 = loss(pred_2, ys2)
        ls = torch.sum(torch.stack([ls1, ls2]))
        ls.backward(retain_graph=True)
        ls1.backward()
        ls2.backward()

        opt.step()
        opt1.step()
        opt2.step()

    for i, data in enumerate(zip(cycle(dataLoader_tr1), cycle(dataLoader_tr2), dataULoader_tr)):
        data_s1 = data[0]
        data_s2 = data[1]
        data_su = data[2]

        xs1 = data_s1[0]
        ys1 = data_s1[1].view(-1, 1)
        xs2 = data_s2[0]
        ys2 = data_s2[1].view(-1, 1)
        xsu = data_su[0]

        fx_1 = model(xs1)
        fx_2 = model(xs2)
        fx_u = model(xsu)

        pred_1 = predict_1(fx_1)
        pred_2 = predict_2(fx_2)
        pred_1u = predict_1(fx_u)
        pred_2u = predict_2(fx_u)

        opt.zero_grad()
        opt1.zero_grad()
        opt2.zero_grad()

        ls1 = loss(pred_1, ys1)
        ls2 = loss(pred_2, ys2)

        coral_loss = args.lam1 * (coral(fx_1, fx_u) + coral(fx_2, fx_u))
        con_loss = args.lam2 * loss(pred_1u, pred_2u)

        ls_opt = torch.sum(torch.stack([ls1, ls2, con_loss, coral_loss]))
        ls_opt1 = torch.sum(torch.stack([ls1, con_loss]))
        ls_opt2 = torch.sum(torch.stack([ls2, con_loss]))

        ls_opt.backward(retain_graph=True)
        ls_opt1.backward(retain_graph=True)
        ls_opt2.backward()

        opt.step()
        opt1.step()
        opt2.step()

        r1_train, _ = pearsonr(pred_1.detach().numpy().flatten(), ys1.detach().numpy().flatten())
        r2_train, _ = pearsonr(pred_2.detach().numpy().flatten(), ys2.detach().numpy().flatten())

        pred_cost.append(ls1.item() + ls2.item())
        coral_cost.append(coral_loss.item())
        con_cost.append(con_loss.item())

        epoch_PR1.append(r1_train)
        epoch_PR2.append(r2_train)

        loss1.append(ls1)
        loss2.append(ls2)

    return np.mean(pred_cost), np.mean(coral_cost), np.mean(con_cost), np.mean(epoch_PR1), np.mean(
        epoch_PR2), torch.mean(torch.stack(loss1)), torch.mean(torch.stack(loss2))


def validate_workflow(args, model, predict_1, predict_2, ws, loss, TX_val_N, Ty_val):
    model.eval()

    w_n = torch.nn.functional.softmax(torch.stack(ws), dim=0)
    w1 = w_n[0]
    w2 = w_n[1]

    fx_val = model(TX_val_N)
    pred_1 = predict_1(fx_val)
    pred_2 = predict_2(fx_val)
    pred_val = w1 * pred_1 + w2 * pred_2
    ls = loss(pred_val, Ty_val.view(-1, 1))

    r_val, _ = pearsonr(pred_val.detach().numpy().flatten(), Ty_val.detach().numpy().flatten())
    # sr_val,_ = spearmanr(pred_val.detach().numpy().flatten(), Ty_val.detach().numpy().flatten())
    # kr_val,_ = kendalltau(pred_val.detach().numpy().flatten(), Ty_val.detach().numpy().flatten())

    return ls.item(), r_val, pred_val


def heldout_test(args, best_model, predict_1, predict_2, ws, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2,
                 scaler):
    w_n = torch.nn.functional.softmax(torch.stack(ws), dim=0)
    w1 = w_n[0]
    w2 = w_n[1]

    X_ts1_N = scaler.transform(X_ts_1)
    X_ts2_N = scaler.transform(X_ts_2)

    TX_ts1_N = torch.FloatTensor(X_ts1_N)
    TX_ts2_N = torch.FloatTensor(X_ts2_N)

    AAC_ts11 = predict_1(best_model(TX_ts1_N))
    AAC_ts12 = predict_2(best_model(TX_ts1_N))
    pred_ts1 = w1 * AAC_ts11 + w2 * AAC_ts12

    AAC_ts21 = predict_1(best_model(TX_ts2_N))
    AAC_ts22 = predict_2(best_model(TX_ts2_N))
    pred_ts2 = w1 * AAC_ts21 + w2 * AAC_ts22

    r_ts1, _ = pearsonr(pred_ts1.detach().numpy().flatten(), Y_ts_1)
    sr_ts1, _ = spearmanr(pred_ts1.detach().numpy().flatten(), Y_ts_1)

    roc_ts2 = roc_auc_score(Y_ts_2.astype(int), pred_ts2.detach().numpy().flatten(),
                            average='micro')
    aupr_ts2 = average_precision_score(Y_ts_2.astype(int), pred_ts2.detach().numpy().flatten(),
                                       average='micro')

    return r_ts1, sr_ts1, roc_ts2, aupr_ts2


def heldout_testv3(args, best_model, predict_1, predict_2, ws, X_ts_1, Y_ts_1, X_ts_2, Y_ts_2,
                   X_ts_3, Y_ts_3, scaler):
    w_n = torch.nn.functional.softmax(torch.stack(ws), dim=None)
    w1 = w_n[0]
    w2 = w_n[1]

    X_ts1_N = scaler.transform(X_ts_1)
    X_ts2_N = scaler.transform(X_ts_2)
    X_ts3_N = scaler.transform(X_ts_3)

    TX_ts1_N = torch.FloatTensor(X_ts1_N)
    TX_ts2_N = torch.FloatTensor(X_ts2_N)
    TX_ts3_N = torch.FloatTensor(X_ts3_N)

    AAC_ts11 = predict_1(best_model(TX_ts1_N))
    AAC_ts12 = predict_2(best_model(TX_ts1_N))
    pred_ts1 = w1 * AAC_ts11 + w2 * AAC_ts12

    AAC_ts21 = predict_1(best_model(TX_ts2_N))
    AAC_ts22 = predict_2(best_model(TX_ts2_N))
    pred_ts2 = w1 * AAC_ts21 + w2 * AAC_ts22

    AAC_ts31 = predict_1(best_model(TX_ts3_N))
    AAC_ts32 = predict_2(best_model(TX_ts3_N))
    pred_ts3 = w1 * AAC_ts31 + w2 * AAC_ts32

    r_ts1, _ = pearsonr(pred_ts1.detach().numpy().flatten(), Y_ts_1)
    sr_ts1, _ = spearmanr(pred_ts1.detach().numpy().flatten(), Y_ts_1)

    roc_ts2 = roc_auc_score(Y_ts_2.astype(int), pred_ts2.detach().numpy().flatten(),
                            average='micro')
    aupr_ts2 = average_precision_score(Y_ts_2.astype(int), pred_ts2.detach().numpy().flatten(),
                                       average='micro')

    roc_ts3 = roc_auc_score(Y_ts_3.astype(int), pred_ts3.detach().numpy().flatten(),
                            average='micro')
    aupr_ts3 = average_precision_score(Y_ts_3.astype(int), pred_ts3.detach().numpy().flatten(),
                                       average='micro')

    return r_ts1, sr_ts1, roc_ts2, aupr_ts2, roc_ts3, aupr_ts3


def heldout_testF(args, best_model, predict_1, predict_2, ws, PRAD_N, KIRC_N, gCSI_N,
                  gCSI_aac_drug):
    w_n = torch.nn.functional.softmax(torch.stack(ws), dim=0)
    w1 = w_n[0]
    w2 = w_n[1]

    AAC_ts1 = predict_1(best_model(gCSI_N))
    AAC_ts2 = predict_2(best_model(gCSI_N))
    pred_gCSI_N = w1 * AAC_ts1 + w2 * AAC_ts2

    AAC_PRAD_N1 = predict_1(best_model(PRAD_N))
    AAC_PRAD_N2 = predict_2(best_model(PRAD_N))
    pred_PRAD_N = w1 * AAC_PRAD_N1 + w2 * AAC_PRAD_N2

    AAC_KIRC_N1 = predict_1(best_model(KIRC_N))
    AAC_KIRC_N2 = predict_2(best_model(KIRC_N))
    pred_KIRC_N = w1 * AAC_KIRC_N1 + w2 * AAC_KIRC_N2

    r_gCSI_N, pval1 = pearsonr(pred_gCSI_N.detach().numpy().flatten(), gCSI_aac_drug.values)
    sr_gCSI_N, pval2 = spearmanr(pred_gCSI_N.detach().numpy().flatten(), gCSI_aac_drug.values)

    return r_gCSI_N, pval1, sr_gCSI_N, pval2, \
           pred_PRAD_N.detach().numpy().flatten(), \
           pred_KIRC_N.detach().numpy().flatten()


def plots(args, train_loss, con_loss, coral_loss, train_pr1, train_pr2, val_loss, val_pr):
    if not os.path.isdir(args.save_results + "/" + "plots/"):
        os.makedirs(args.save_results + "/" + "plots/")
    new_path = args.save_results + "/" + "plots/"

    title_loss = 'Total loss Train drug ={}, batch_size = {}, lr = ({},{},{}), epoch = {}, hd = {}. wd = ({},{},{}), ldr = {}'. \
        format(args.drug, args.bs, args.lr, args.lr1, args.lr2, args.epoch, args.hd, args.wd,
               args.wd1, args.wd2, args.ldr)
    plt.plot(np.squeeze(train_loss), '-r')
    plt.title(title_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(os.path.join(new_path, title_loss + '.png'), dpi=300)
    plt.close()

    title_loss = 'Total Con loss Train drug ={}, batch_size = {}, lr = ({},{},{}), epoch = {}, hd = {}. wd = ({},{},{}), ldr = {}'. \
        format(args.drug, args.bs, args.lr, args.lr1, args.lr2, args.epoch, args.hd, args.wd,
               args.wd1, args.wd2, args.ldr)
    plt.plot(np.squeeze(con_loss), '-r')
    plt.title(title_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(os.path.join(new_path, title_loss + '.png'), dpi=300)
    plt.close()

    title_loss = 'Total Coral loss Train drug ={}, batch_size = {}, lr = ({},{},{}), epoch = {}, hd = {}. wd = ({},{},{}), ldr = {}'. \
        format(args.drug, args.bs, args.lr, args.lr1, args.lr2, args.epoch, args.hd, args.wd,
               args.wd1, args.wd2, args.ldr)
    plt.plot(np.squeeze(coral_loss), '-r')
    plt.title(title_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(os.path.join(new_path, title_loss + '.png'), dpi=300)
    plt.close()

    title_PR = 'Correlation Train 1 drug ={}, batch_size = {}, lr = ({},{},{}), epoch = {}, hd = {}. wd = ({},{},{}), ldr = {}'. \
        format(args.drug, args.bs, args.lr, args.lr1, args.lr2, args.epoch, args.hd, args.wd,
               args.wd1, args.wd2, args.ldr)
    plt.plot(np.squeeze(train_pr1), '-r', np.squeeze(val_pr), '-b')
    plt.title(title_PR)
    plt.ylabel('Pearson')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(new_path, title_PR + '.png'), dpi=300)
    plt.close()

    title_PR = 'Correlation Train 2 drug = {}, batch_size = {}, lr = ({},{},{}), epoch = {}, hd = {}. wd = ({},{},{}), ldr = {}'. \
        format(args.drug, args.bs, args.lr, args.lr1, args.lr2, args.epoch, args.hd, args.wd,
               args.wd1, args.wd2, args.ldr)
    plt.plot(np.squeeze(train_pr2), '-r', np.squeeze(val_pr), '-b')
    plt.title(title_PR)
    plt.ylabel('Pearson')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(new_path, title_PR + '.png'), dpi=300)
    plt.close()


def coral(source, target):
    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    #     loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).reshape(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


def set_best_params_from_paper(args):
    if args.best_params:  # best parameters from paper
        print('Using best parameters from the paper for %s' % args.drug)

        out_dir = 'VelodromeTrain' if args.stage == 'train' else 'VelodromeTest'
        if args.drug == "Docetaxel":
            args.save_logs = './%s/%s/logs/' % (out_dir, args.drug)
            args.save_models = './%s/%s/models/' % (out_dir, args.drug)
            args.save_results = './%s/%s/results/' % (out_dir, args.drug)
            args.seed = 42
            args.epoch = 10
            args.bs = 65
            args.ldr = 0.1
            args.wd = 0.05
            args.wd1 = 0.0005
            args.wd2 = 0.0001
            args.hd = 3
            args.lr = 0.001
            args.lr1 = 0.005
            args.lr2 = 0.0005
            args.lam1 = 0.2
            args.lam2 = 0.8

        elif args.drug == "Gemcitabine":
            args.save_logs = './%s/%s/logs/' % (out_dir, args.drug)
            args.save_models = './%s/%s/models/' % (out_dir, args.drug)
            args.save_results = './%s/%s/results/' % (out_dir, args.drug)
            args.seed = 5376
            args.epoch = 10
            args.bs = 17
            args.ldr = 0.1
            args.wd = 0.0001
            args.wd1 = 0.005
            args.wd2 = 0.01
            args.hd = 2
            args.lr = 0.01
            args.lr1 = 0.005
            args.lr2 = 0.05
            args.lam1 = 0.005
            args.lam2 = 0.99

        elif args.drug == "Erlotinib":
            args.save_logs = './%s/%s/logs/' % (out_dir, args.drug)
            args.save_models = './%s/%s/models/' % (out_dir, args.drug)
            args.save_results = './%s/%s/results/' % (out_dir, args.drug)
            args.seed = 42
            args.epoch = 50
            args.bs = 129
            args.ldr = 0.1
            args.wd = 0.05
            args.wd1 = 0.005
            args.wd2 = 0.0005
            args.hd = 2
            args.lr = 0.001
            args.lr1 = 0.01
            args.lr2 = 0.001
            args.lam1 = 0.01
            args.lam2 = 0.99

        elif args.drug == "Paclitaxel":
            args.save_logs = './%s/%s/logs/' % (out_dir, args.drug)
            args.save_models = './%s/%s/models/' % (out_dir, args.drug)
            args.save_results = './%s/%s/results/' % (out_dir, args.drug)
            args.seed = 42
            args.epoch = 50
            args.bs = 129
            args.ldr = 0.1
            args.wd = 0.005
            args.wd1 = 0.05
            args.wd2 = 0.005
            args.hd = 2
            args.lr = 0.05
            args.lr1 = 0.0005
            args.lr2 = 0.0001
            args.lam1 = 0.3
            args.lam2 = 0.7
