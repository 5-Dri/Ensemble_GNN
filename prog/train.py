import hydra
from hydra import utils
from omegaconf import DictConfig
import mlflow
import torch

from train_planetoid  import run as train_planetoid
# from train_arxiv      import run as train_arxiv
# from train_ppi        import run as train_ppi
# from train_ppi_induct import run as train_ppi_induct
from utils import fix_seed, log_params_from_omegaconf_dict, log_artifacts



@hydra.main(version_base="1.1", config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg_mlflow = cfg.mlflow
    cfg = cfg[cfg.key]
    root = utils.get_original_cwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fix_seed(cfg.seed)
    mlflow.set_tracking_uri('http://' + cfg_mlflow.server_ip + ':5000')
    mlflow.set_experiment(cfg_mlflow.runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        if cfg.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            valid_acces, test_acces, artifacts = train_planetoid(cfg, root, device)
        # elif cfg.dataset == 'Arxiv':
        #     valid_acces, test_acces, artifacts = train_arxiv(cfg, root, device)
        # elif cfg.dataset == 'PPI':
        #     valid_acces, test_acces, artifacts = train_ppi(cfg, root, device)
        # elif cfg.dataset == 'PPIinduct':
        #     valid_acces, test_acces, artifacts = train_ppi_induct(cfg, root, device)

        for i, acc_test in enumerate(test_acces):
            mlflow.log_metric('acc_test', value=acc_test, step=i)
        valid_acc_mean = sum(valid_acces)/len(valid_acces)
        test_acc_mean = sum(test_acces)/len(test_acces)
        mlflow.log_metric('acc_mean', value=test_acc_mean)
        mlflow.log_metric('acc_max', value=max(test_acces))
        mlflow.log_metric('acc_min', value=min(test_acces))
        mlflow.log_metric('valid_mean', value=valid_acc_mean)
        # log_artifacts(artifacts)

    print('valid mean acc: {:.3f}'.format(valid_acc_mean))
    print('test  mean acc: {:.3f}'.format(test_acc_mean))
    return valid_acc_mean # we tune hyper-parameters based on validation data, not test data


if __name__ == "__main__":
    main()