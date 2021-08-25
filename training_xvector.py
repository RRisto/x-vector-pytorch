from x_vectors.trainer import Trainer, Args

args=Args()
args.get_args()
args.loss_fun='AngleLoss'
# args.loss_fun='CrossEntropyLoss'
# args.loss_fun='AngularLoss'
# args.loss_fun='AdMSoftmaxLoss'
args.batch_size=3

trainer=Trainer(args)

#train
trainer.train(10)