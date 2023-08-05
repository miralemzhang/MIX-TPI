import argparse
import json
from dataset import *
from train import *
from network.model import MIX_TPI
from torch import nn

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def parse_input(description: str) -> argparse.ArgumentParser:
    """ parsing input arguments
     """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--split_mode",
        default="tcr_split",
        help="split mode, [tcr_split, strict_split]",
        type=str,
    )
    p.add_argument(
        "--data_path",
        default="data/TITAN/ten_fold/",
        help="path of data",
        type=str,
    )
    p.add_argument(
        "--features",
        default="hydrophob,isoelectric,mass,hydrophil",
        help="A string of comma separated values listed in peptide_feature.featuresMap.",
        required=False,
        type=str,
    )
    p.add_argument(
        "--dataset",
        default="TITAN",
        help="TITAN, TITAN-covid",
        required=False,
        type=str,
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    args = parse_input("MIX-TPI")

    with open("./params/MIX-TPI.json", "r") as f:
        param = json.load(f)
    for key, val in param.items():
        logging.info("argument %s: %r", key, val)
    for arg, value in sorted(vars(args).items()):
        logging.info("argument %s: %r", arg, value)

    dataset = args.data_path.split("/")[1]
    train_auc_result = []
    train_aupr_result = []
    train_loss_result = []
    test_auc_result = []
    test_aupr_result = []
    test_loss_result = []
    for fold in range(10):
        logging.info(f"----------------fold{fold} start.----------------")
        if args.dataset == "TITAN":
            train_data_path = args.data_path + args.split_mode + f"/fold{fold}/train.csv"
            test_data_path = args.data_path + args.split_mode + f"/fold{fold}/test.csv"
        else:
            train_data_path = args.data_path + args.split_mode + f"/fold{fold}/train+covid.csv"
            test_data_path = args.data_path + args.split_mode + f"/fold{fold}/test+covid.csv"
        embed_output_dir = f"ckpt/TITAN/{args.split_mode}/fold{fold}/"
        if not os.path.exists(embed_output_dir):
            os.makedirs(embed_output_dir)

        train_loader = DataLoader(
            read_data(train_data_path),
            param["batch_size"],
            args,
            param["tcr_padding_length"],
            param["peptide_padding_length"],
        )

        # Load test set with batchifing and tokenizing.
        test_loader = DataLoader(
            read_data(test_data_path),
            param["batch_size"],
            args,
            param["tcr_padding_length"],
            param["peptide_padding_length"],
            save_dir=embed_output_dir + 'test.csv'
        )

        logging.info(
            "Dataset: {}, Train set num: {}, Test set num: {}".format(
                dataset, len(train_loader), len(test_loader)
            )
        )

        model = MIX_TPI(
            tcr_padding_len=param["tcr_padding_length"],
            peptide_padding_len=param["peptide_padding_length"],
            embedding_dim=param["embedding_dim"],
            map_num=len(args.features.split(',')),
            dropout_prob=param["dropout"],
            n_head=param["n_head"],
            gating=param["gating"],
            hidden_channel=param["hidden_channel"],
            k=param["k"]
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=param["lr"])

        criterion = nn.BCELoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            logger=logging.getLogger(__name__),
            save_dir=embed_output_dir,
            patience=10
        )

        fold_train_loss, fold_test_loss, fold_train_auc, fold_test_auc, fold_train_aupr, fold_test_aupr = trainer.train(
            train_loader=train_loader, test_loader=test_loader, epochs=param["epoch"]
        )

        # np.savez_compressed(embed_output_dir+'array_ckpt.npz', X_train_all=X_train_all.numpy(), X_test=X_test.numpy())

        train_auc_result.append(fold_train_auc)
        train_aupr_result.append(fold_train_aupr)
        train_loss_result.append(fold_train_loss)
        test_auc_result.append(fold_test_auc)
        test_aupr_result.append(fold_test_aupr)
        test_loss_result.append(fold_test_loss)

    logging.info(
        "\nfinish!!\n train loss: %.4f±%.4f, test loss: %.4f±%.4f \n train auc: %.4f±%.4f, test auc: %.4f±%.4f \n train aupr: %.4f±%.4f, test aupr: %.4f±%.4f" % (
            np.mean(train_loss_result), np.std(train_loss_result), np.mean(test_loss_result), np.std(test_loss_result),
            np.mean(train_auc_result), np.std(train_auc_result), np.mean(test_auc_result), np.std(test_auc_result),
            np.mean(train_aupr_result), np.std(train_aupr_result), np.mean(test_aupr_result), np.std(test_aupr_result),
        )
    )

if __name__ == "__main__":
    main()
