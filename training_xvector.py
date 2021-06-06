from x_vectors.trainer import Trainer, Args

args=Args()
args.get_args()

trainer=Trainer(args)

#train
trainer.train(3)