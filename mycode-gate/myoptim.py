from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from fairseq.optim import register_optimizer
from omegaconf import II, DictConfig

@register_optimizer("myadam", dataclass=FairseqAdamConfig)
class MyFairseqAdam(FairseqAdam):
	def __init__(self, cfg: DictConfig, params):
		super().__init__(cfg, params)
		# group_1, group_2=[],[]
		# for name, p in params:
		# 	if 'attacker' in name: 
		# 		group_1.append(p)
		# 	else:
		# 		group_2.append(p)
		# self._optimizer = [super(Adam, self).__init__(group_1, defaults), 
		# super(Adam, self).__init__(group_2, defaults)]

	def backward(self, loss, retain_graph=False):
		loss.backward(retain_graph=retain_graph)
		# for p in self.params:
		# 	if 'attacker' in name: 
		# 		p.requires_grad = False
		# 	else:
		# 		p.requires_grad = True
		# loss1.backward()
		# for p in self.params:
		# 	if 'attacker' in name: 
		# 		p.requires_grad = True
		# 	else:
		# 		p.requires_grad = False
		# loss2.backward()
	
	# def step(self, closure=None, scale=1.0):
	# 	self._optimizer[0].step()
	# 	self._optimizer[1].step()
	# 	return 



