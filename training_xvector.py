from x_vectors.trainer import Trainer, Args

args=Args()
args.get_args()
args.loss_fun='AngleLoss'
args.loss_fun='CrossEntropyLoss'

trainer=Trainer(args)

#train
trainer.train(10)