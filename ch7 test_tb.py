from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

writer.add_scalar()

writer.close()
