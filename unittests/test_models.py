import pytest

import numpy as np

from src.models.models import SimpleCNN, SimpleFeedForward
from src.models.lightning_models import LightningRegression, LightningClassification

import torch
import torch.utils.data
import pytorch_lightning as pl
from torchvision import transforms

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

pl.seed_everything(42)

class TestLightningClassification:

    def test_feedfoward(self):

        n_classes = 3
        final_activation = 'softmax'
        opt = torch.optim.Adam
        opt_params = {'lr': 0.001}
        loss = torch.nn.CrossEntropyLoss()
        ckpt_path = 'unittests/testoutput/ckpts'

        X, y = make_classification(n_samples=1000, n_features=40, n_informative=25, n_classes=n_classes, n_clusters_per_class=1)
        subs = np.ones_like(y)
        X_train, X_test, y_train, y_test, subs_train, subs_test = train_test_split(X, y, subs, test_size=0.2)

        clf = SVC()
        clf.fit(X_train, y_train)
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)

        X_train, X_test, y_train, y_test, subs_train, subs_test = (torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float),
                                                                   torch.tensor(y_train, dtype=torch.long), torch.tensor (y_test, dtype=torch.long),
                                                                   torch.tensor(subs_train, dtype=torch.long), torch.tensor (subs_test, dtype=torch.long))
        input_shape = X_train.shape[1:]

        # make dataloader here
        ds_train = torch.utils.data.TensorDataset(X_train , y_train, subs_train)
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [int(len(ds_train) * 0.9), int(len(ds_train) * 0.1)])
        ds_test = torch.utils.data.TensorDataset(X_test, y_test, subs_test)

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=32, shuffle=True)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)

        model = SimpleFeedForward(input_shape=input_shape,
                                n_classes=n_classes,
                                units=[128, 256, 128],
                                dropout=[0.1, 0.1, 0.1])

        net = LightningClassification(model=model,
                                      final_activation=final_activation,
                                      optimizer=opt,
                                      optimizer_params=opt_params,
                                      loss_fn=loss)

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode='min', monitor='val_loss'
            ),
           pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        trainer = pl.Trainer(default_root_dir=ckpt_path,
                             callbacks=callbacks,
                             max_epochs=50,
                             gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(net, dl_train, dl_val)

        net = LightningClassification.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
            final_activation=final_activation,
            optimizer=opt,
            optimizer_params=opt_params,
            loss_fn=loss
        )

        train_result = trainer.test(net, dataloaders=dl_train, verbose=True)
        val_result = trainer.test(net, dataloaders=dl_val, verbose=True)
        test_result = trainer.test(net, dataloaders=dl_test, verbose=True)

        assert train_result[0]['test_acc'] == pytest.approx(.99, abs=0.03)
        assert val_result[0]['test_acc'] == pytest.approx(.92, abs=0.05)
        assert test_result[0]['test_acc'] == pytest.approx(.92, abs=0.05)


    def test_cnn(self):
        """
        Note: This Test should be run on a GPU otherwise it may take around 40 minutes to run
        """
        from torchvision.datasets import CIFAR10
        from torch.utils.data import default_collate
        ds_train = CIFAR10('unittests/testdata', train=True, download=True, transform=transforms.ToTensor())
        ds_test = CIFAR10('unittests/testdata', train=False, download=True, transform=transforms.ToTensor())

        n_classes = len(ds_train.classes)
        final_activation = 'softmax'
        opt = torch.optim.Adam
        opt_params = {'lr': 1e-4}
        loss = torch.nn.CrossEntropyLoss()
        ckpt_path = 'unittests/testoutput/ckpts'

        input_shape = (3, 32, 32)

        # make dataloader here
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [int(len(ds_train) * 0.9), int(len(ds_train) * 0.1)])

        # collate function to add third element to batch
        def dummy_collate(data):
            batch = default_collate(data)
            subs = torch.zeros(len(batch[0]))
            batch.append(subs)
            return batch

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=dummy_collate)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=32, shuffle=True, collate_fn=dummy_collate)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, collate_fn=dummy_collate)

        # simple model from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
        # should get around 67% accuracy around 40 epochs
        model = SimpleCNN(input_shape=input_shape,
                          n_classes=n_classes,
                          channels=[32, 32],
                          kernels=[3, 3],
                          pools=[(1, 1), (2, 2)],
                          cnn_dropout=[0.2, 0.2],
                          fc_units=[128],
                          fc_dropout=[0.3])

        net = LightningClassification(model=model,
                                      final_activation=final_activation,
                                      optimizer=opt,
                                      optimizer_params=opt_params,
                                      loss_fn=loss)

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode='min', monitor='val_loss'
            ),
        ]
        trainer = pl.Trainer(default_root_dir=ckpt_path,
                             callbacks=callbacks,
                             max_epochs=10,
                             gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(net, dl_train, dl_val)

        net = LightningClassification.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=model,
            final_activation=final_activation,
            optimizer=opt,
            optimizer_params=opt_params,
            loss_fn=loss
        )

        train_result = trainer.test(net, dataloaders=dl_train, verbose=True)
        val_result = trainer.test(net, dataloaders=dl_val, verbose=True)
        test_result = trainer.test(net, dataloaders=dl_test, verbose=True)

        assert train_result[0]['test_acc'] == pytest.approx(.70, abs=0.03)
        assert val_result[0]['test_acc'] == pytest.approx(.65, abs=0.03)
        assert test_result[0]['test_acc'] == pytest.approx(.65, abs=0.03)